#!/usr/bin/env python3

# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

# Requires:
#    sympy               (pip3 install sympy)
import sympy
import math
import numpy

########################################################################
# 3GPP 38.211, Section 5.1
def BPSK(b, A):
    return ((1 - 2 * b[0]) + 1j * (1 - 2 * b[0])) * A

def QPSK(b, A):
    return ((1 - 2 * b[0]) + 1j * (1 - 2 * b[1])) * A

def QAM16(b, A):
    return ((1 - 2 * b[0]) * (2 - (1 - 2 * b[2])) + 1j * (1 - 2 * b[1]) * (2 - (1 - 2 * b[3]))) * A

def QAM64(b, A):
    return ((1 - 2*b[0]) * (4 - (1 - 2*b[2]) * (2 - (1 - 2*b[4]))) + 1j * (1 - 2*b[1]) * (4 - (1 - 2*b[3]) * (2 - (1 - 2*b[5])))) * A

def QAM256(b, A):
    return ((1 - 2*b[0]) * (8 - (1 - 2*b[2]) * (4 - (1 - 2*b[4]) * (2 - (1 - 2*b[6])))) + 1j * (1 - 2*b[1]) * (8 - (1 - 2*b[3]) * (4 - (1 - 2*b[5]) * (2 - (1 - 2*b[7]))))) * A

#-----------------------------------------------------------------------
# find_closest_modulation()
# Input:
#     mod_list: list of tuples in which:
#         t[1] is a list of 0 or 1 values
#         t[2] is the modulation axis position (normalized by A)
#     mod_pos: search position (normalized by A)
#     bit_idx: bit index to search
#     bit_value: bit value (0 or 1) for which to find the "closest" to
#                mod_pos
# Output:
#     mod_list item that is the "closes" to mod_pos
def find_closest_modulation(mod_list, mod_pos, bit_idx, bit_value):
    # Extract modulation positions that match for bit b
    mods_with_matching_bit = [m for m in mod_list if m[1][bit_idx] == bit_value]
    # Find the modulation with the minimum distance 
    dist                   = [abs(mod_pos - m[2]) for m in mods_with_matching_bit]
    idx                    = dist.index(min(dist))
    #print('closest mod with bit %d = %d to %d is %d' % (bit_idx, bit_value, mod_pos, mods_with_matching_bit[idx][2]))
    return mods_with_matching_bit[idx]

def value_to_fp16_hex_string(val):
    a = numpy.array([val], dtype = numpy.float16)
    b = a.tobytes()
    return '0x' + format(b[1], '02X') + format(b[0], '02X') + ' /* =%7.4f */' % a[0]
#-----------------------------------------------------------------------
# gen_qam_table()
# q_int: integer QAM (e.g. 4, 16, 64, or 256)
# qam_fn: QAM modulation function, a function of b[] and A
# A_inv_squared: inverse of normalization factor (A), squared. In
#                particular:
#                BPSK:     2
#                QPSK:     2
#                QAM16:   10
#                QAM64:   42
#                QAM256: 170
def gen_QAM_table(qam_int, qam_fn, A_inv_squared, f_h):
    #- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    # A: QAM normalization factor used for unity average
    #    symbol energy
    A       = sympy.symbols('A', real=True)
    # b: message bits for modulation/demodulation
    b       = sympy.IndexedBase('b', shape=(8))
    #- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    # Symbolic (sympy) representation of the modulation function
    sQAM      = qam_fn(b, A)    # sympy expression
    sQAM_re   = sympy.re(sQAM)  # real part only
    sQAM_norm = sQAM_re / A     # normalized by A
    #- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    # Calculate modulation values spanned by real bits only - the
    # imaginary values are identical.
    QAM_bits = int(math.log(qam_int, 2))
    # Special case PAM_bits for BPSK here:
    PAM_bits = max(int(QAM_bits / 2), 1)
    #print('QAM_bits = %i, PAM_bits = %i' % (QAM_bits, PAM_bits)) 
    # Generate a list of tuples for the modulation values:
    # (string_representation, list_representation, mod_value_normalized_by_A)
    mods_unsorted = []
    for i in range(int(math.pow(2, PAM_bits))):
        bin_string = ('{0:0%db}' % PAM_bits).format(i)
        # Note order reversal here: '0010' ---> [0, 1, 0, 0]
        bin_list   = [int(bin_string[len(bin_string) - 1 -j]) for j in range(len(bin_string))]
        #print(bin_string, bin_list)
        # Set all bit values to 0
        subs_list = [(b[j], 0) for j in range(QAM_bits)]
        for j in range(PAM_bits):
            # Change sympy 'substitute' values for bits that are set to 1
            if '1' == bin_string[len(bin_string) - 1 - j]:
                subs_list[j*2] = (b[j*2], 1)
                #print('Setting b[%d] to 1 for %s' % (j*2, bin_string))
        mod_val = int(sQAM_norm.subs(subs_list))
        #print(type(val))
        mods_unsorted.append((bin_string, bin_list, mod_val))
        #print(bin_string, sQAM_re.subs(subs_list))
    #- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    # Sort values by position on the real axis. Result should indicate
    # a gray code.
    # mods tuple:
    # (string_representation, list_representation, mod_value_normalized_by_A)
    mods = sorted(mods_unsorted, key = lambda t: t[2])
    #print(mods)
    print('QAM%d (PAM%d)' % (qam_int, int(math.pow(2, PAM_bits))))
    for m in mods:
        print('    %s: %3d A' % (m[0], m[2]))
    #- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    # Determine the size of the table to export. We currently add 1 to
    # the number of PAM bits and add extra padding on each size of the
    # desired range.
    N_TABLE = int(math.pow(2, PAM_bits + 1))
    # We will have a sample at 0. For an even number of samples, we
    # arbitrarily place the extra in the negative values.
    min_value_A = -N_TABLE
    Zr_A_values = [min_value_A + (2 * idx) for idx in range(N_TABLE)]
    A_val       = 1 / math.sqrt(A_inv_squared)
    Zr_values   = [Zr_A * A_val for Zr_A in Zr_A_values]
    #print(Zr_A_values)
    #- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    LLR_table = []
    for b in range(PAM_bits):
        print('    Bit %d:' % b)
        LLR_bit_values = []
        for idx, Zr_A in enumerate(Zr_A_values):
            x0_A = find_closest_modulation(mods, Zr_A, b, 0)[2]
            x1_A = find_closest_modulation(mods, Zr_A, b, 1)[2]
            s    = ((Zr_A - x1_A)**2 - (Zr_A - x0_A)**2) / 2
            print('        [%2d] %3dA: x_0 = %3dA, x1 = %3dA, s = %4g A^2 / sigma^2' % (idx, Zr_A, x0_A, x1_A, s))
            #LLR_bit_values.append(s * (A_val**2))
            LLR_bit_values.append(s / A_inv_squared)
        LLR_table.append(LLR_bit_values)
        #print(LLR_table)
    #- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    # Add half precision constants to the soft demapper tables for use
    # by CUDA kernels
    var_name = 'QAM_%d_table' % qam_int
    f_h.write('\nconst __half_raw %s[] = \n{\n' % var_name)
    for row in range(N_TABLE):
        f_h.write('    /* [%2d: %3dA] */ ' % (row, Zr_A_values[row]))
        # We will write 4 values to make the resulting tables uniform
        # for all QAMs
        for col in range(4):
            val = 0.0 if col >= PAM_bits else LLR_table[col][row]
            val_str = value_to_fp16_hex_string(val)
            f_h.write(val_str)
            if col != 3:
                f_h.write(', ')
        if row != N_TABLE - 1:
            f_h.write(',')
        f_h.write('\n')
    f_h.write('};\n')
    #- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    # Write a text table that will be used by the MATLAB test script
    fname = 'QAM%d_LLR.txt' % qam_int
    print('Writing %s' % fname)
    f = open(fname, 'w')
    f.write('Zr')
    for b in range(PAM_bits):
        f.write(',bit%d' %b)
    f.write('\n')
    for idx, Zr in enumerate(Zr_values):
        f.write('%f' % Zr)
        for b in range(PAM_bits):
            f.write(',%f' % LLR_table[b][idx])
        f.write('\n')
    f.close()

#**********************************************************************
# Linear mapping (y = mx + b) from symbol value (x), representing the
# in-phase or quadrature component, to a normalized texture coordinate
# in the range of (0..1).
# Texture tables are generated for symbol values from -NA to (N-2)A
# (where N is the table size and A is the QAM normalization factor)
# with a spacing of 2A.
# We want symbol value -NA to map to texture coordinate 1/(2N), which
# is the texture coordinate of the first provided sample value.
# We want (N-2)A to map to texture coordinate 1 - 1/(2N), which is
# the texture coordinate of the last provided sample value.
def tex_coord_transform_slope(N, A):
    return 1.0 / (2.0 * N * A)

def tex_coord_transform_intercept(N):
    return 0.5  + (1.0 /  (2.0 * N))

def gen_half_tex_coord_transform_table(f):
    norm_factors = {1: 1.0 / math.sqrt(2),
                    2: 1.0 / math.sqrt(2),
                    4: 1.0 / math.sqrt(10),
                    6: 1.0 / math.sqrt(42),
                    8: 1.0 / math.sqrt(170)}
    
    f.write("""\n\nstruct mod_symbol_half_to_tex_coord_t
{
    __half2_raw m;
    __half2_raw b;
};

__constant__ mod_symbol_half_to_tex_coord_t sym_transform_h[9] =
{
""")
    for idx in range(9):
        QAM_bits = idx
        m = 0.0
        b = 0.0
        if idx in norm_factors:
            PAM_bits = max(int(QAM_bits / 2), 1)
            N = int(math.pow(2, PAM_bits + 1))
            m = tex_coord_transform_slope(N, norm_factors[idx])
            b = tex_coord_transform_intercept(N)
            print('idx = %i, N = %i, A = %f, m = %f, b = %f' % (idx, N, norm_factors[idx], m, b))
        m_str = value_to_fp16_hex_string(m)
        b_str = value_to_fp16_hex_string(b)
        f.write('    {  {%s, %s} , {%s, %s} }, // %i\n' % (m_str, m_str, b_str, b_str, idx))
    f.write('};\n\n')

fname_h = 'soft_demapper_tables.h'
f_h     = open(fname_h, 'w')
nv_copyright = """/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""
f_h.write(nv_copyright)
f_h.write('\n\n#if !defined(SOFT_DEMAPPER_TABLES_H_INCLUDED_)\n')
f_h.write('#define SOFT_DEMAPPER_TABLES_H_INCLUDED_\n\n')

f_h.write('#include <cuda_fp16.h>\n\n')
#gen_QAM_table(2,   BPSK,   1 / math.sqrt(2),   f_h)
gen_QAM_table(4,   QPSK,   2,   f_h)
gen_QAM_table(16,  QAM16,  10,  f_h)
gen_QAM_table(64,  QAM64,  42,  f_h)
gen_QAM_table(256, QAM256, 170, f_h)

f_h.write('\n\n#endif // !defined(SOFT_DEMAPPER_TABLES_H_INCLUDED_)\n\n')
print('Writing %s' % fname_h)
f_h.close()

fname_cuh = 'soft_demapper_tables.cuh'
f_cuh     = open(fname_cuh, 'w')
nv_copyright = """/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""
f_cuh.write(nv_copyright)
f_cuh.write('\n\n#if !defined(SOFT_DEMAPPER_TABLES_CUH_INCLUDED_)\n\n')
f_cuh.write('#define SOFT_DEMAPPER_TABLES_CUH_INCLUDED_\n\n')

f_cuh.write('#include <cuda_fp16.h>\n\n')
f_cuh.write('namespace soft_demapper\n{\n')

gen_half_tex_coord_transform_table(f_cuh)
            
f_cuh.write('} // namespace soft_demapper\n')
            
f_cuh.write('\n\n#endif // !defined(SOFT_DEMAPPER_TABLES_CUH_INCLUDED_)\n\n')
print('Writing %s' % fname_cuh)
f_cuh.close()
