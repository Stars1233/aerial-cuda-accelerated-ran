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

// Note: This file has been automatically generated.
#ifndef NRLDPC_TEMPLATES_CUH_INCLUDED_
#define NRLDPC_TEMPLATES_CUH_INCLUDED_ 1
#include <stdint.h>
#include <type_traits>

namespace ldpc2
{
////////////////////////////////////////////////////////////////////////
// max_row_degree
// Provides the maximum row degree for check nodes as a function of
// base graph index (1 or 2)
template <int BG> struct max_row_degree;
template<> struct max_row_degree<1> { static const int value = 19; };
template<> struct max_row_degree<2> { static const int value = 10; };

////////////////////////////////////////////////////////////////////////
// max_info_nodes
// Provides the maximum number of info (or non-parity) nodes as a function of
// base graph index (1 or 2). Note that for BG2, the number of information nodes
// can be less that the maximum.
template <int BG> struct max_info_nodes;
template<> struct max_info_nodes<1> { static const int value = 22; };
template<> struct max_info_nodes<2> { static const int value = 10; };

////////////////////////////////////////////////////////////////////////
// max_parity_nodes
// Provides the maximum number of parity nodes as a function of base graph
// index (1 or 2).
template <int BG> struct max_parity_nodes;
template<> struct max_parity_nodes<1> { static const int value = 46; };
template<> struct max_parity_nodes<2> { static const int value = 42; };

////////////////////////////////////////////////////////////////////////
// max_variable_nodes
// Provides the maximum number of variable nodes as a function of base graph
// index (1 or 2).
template <int BG> struct max_variable_nodes;
template<> struct max_variable_nodes<1> { static const int value = 68; };
template<> struct max_variable_nodes<2> { static const int value = 52; };

////////////////////////////////////////////////////////////////////////
// num_kernel_nodes
// Provides the number of "kernel" nodes as a function of
// base graph index (1 or 2). The number of kernel nodes is
// the number of "systematic" or "information" nodes plus
// the square matrix with a bidiagonal structure.
// See:
// "Algebra-Assisted Construction of Quasi-Cyclic LDPC Codes for
// 5G New Radio," H. Li, B. Bai, X. Mu, J. Zhang, and H. Xu. IEEE
// Access, vol. 6, pp. 50229-50244, 2018.
template <int BG> struct num_kernel_nodes;
template<> struct num_kernel_nodes<1> { static const int value = 26; };
template<> struct num_kernel_nodes<2> { static const int value = 14; };

////////////////////////////////////////////////////////////////////////
// set_index
// Provides the set index for a given lifting size Z
// See Table 5.3.2-1, 3GPP 38.212
template <int Z> struct set_index;
// iLS = 0
template <> struct set_index<2> { static const int value = 0; };
template <> struct set_index<4> { static const int value = 0; };
template <> struct set_index<8> { static const int value = 0; };
template <> struct set_index<16> { static const int value = 0; };
template <> struct set_index<32> { static const int value = 0; };
template <> struct set_index<64> { static const int value = 0; };
template <> struct set_index<128> { static const int value = 0; };
template <> struct set_index<256> { static const int value = 0; };
// iLS = 1
template <> struct set_index<3> { static const int value = 1; };
template <> struct set_index<6> { static const int value = 1; };
template <> struct set_index<12> { static const int value = 1; };
template <> struct set_index<24> { static const int value = 1; };
template <> struct set_index<48> { static const int value = 1; };
template <> struct set_index<96> { static const int value = 1; };
template <> struct set_index<192> { static const int value = 1; };
template <> struct set_index<384> { static const int value = 1; };
// iLS = 2
template <> struct set_index<5> { static const int value = 2; };
template <> struct set_index<10> { static const int value = 2; };
template <> struct set_index<20> { static const int value = 2; };
template <> struct set_index<40> { static const int value = 2; };
template <> struct set_index<80> { static const int value = 2; };
template <> struct set_index<160> { static const int value = 2; };
template <> struct set_index<320> { static const int value = 2; };
// iLS = 3
template <> struct set_index<7> { static const int value = 3; };
template <> struct set_index<14> { static const int value = 3; };
template <> struct set_index<28> { static const int value = 3; };
template <> struct set_index<56> { static const int value = 3; };
template <> struct set_index<112> { static const int value = 3; };
template <> struct set_index<224> { static const int value = 3; };
// iLS = 4
template <> struct set_index<9> { static const int value = 4; };
template <> struct set_index<18> { static const int value = 4; };
template <> struct set_index<36> { static const int value = 4; };
template <> struct set_index<72> { static const int value = 4; };
template <> struct set_index<144> { static const int value = 4; };
template <> struct set_index<288> { static const int value = 4; };
// iLS = 5
template <> struct set_index<11> { static const int value = 5; };
template <> struct set_index<22> { static const int value = 5; };
template <> struct set_index<44> { static const int value = 5; };
template <> struct set_index<88> { static const int value = 5; };
template <> struct set_index<176> { static const int value = 5; };
template <> struct set_index<352> { static const int value = 5; };
// iLS = 6
template <> struct set_index<13> { static const int value = 6; };
template <> struct set_index<26> { static const int value = 6; };
template <> struct set_index<52> { static const int value = 6; };
template <> struct set_index<104> { static const int value = 6; };
template <> struct set_index<208> { static const int value = 6; };
// iLS = 7
template <> struct set_index<15> { static const int value = 7; };
template <> struct set_index<30> { static const int value = 7; };
template <> struct set_index<60> { static const int value = 7; };
template <> struct set_index<120> { static const int value = 7; };
template <> struct set_index<240> { static const int value = 7; };

////////////////////////////////////////////////////////////////////////
// row_degree
// Provides degree (number of nonzero shifted permutation matrices) as
// a function of base graph and check node row.
// See Tables 5.3.2-2 and 5.3.2-3, 3GPP 38.212
template <int BG, int ROW_INDEX> struct row_degree;
template <> struct row_degree<1, 0> { static const int value = 19; };
template <> struct row_degree<1, 1> { static const int value = 19; };
template <> struct row_degree<1, 2> { static const int value = 19; };
template <> struct row_degree<1, 3> { static const int value = 19; };
template <> struct row_degree<1, 4> { static const int value = 3; };
template <> struct row_degree<1, 5> { static const int value = 8; };
template <> struct row_degree<1, 6> { static const int value = 9; };
template <> struct row_degree<1, 7> { static const int value = 7; };
template <> struct row_degree<1, 8> { static const int value = 10; };
template <> struct row_degree<1, 9> { static const int value = 9; };
template <> struct row_degree<1, 10> { static const int value = 7; };
template <> struct row_degree<1, 11> { static const int value = 8; };
template <> struct row_degree<1, 12> { static const int value = 7; };
template <> struct row_degree<1, 13> { static const int value = 6; };
template <> struct row_degree<1, 14> { static const int value = 7; };
template <> struct row_degree<1, 15> { static const int value = 7; };
template <> struct row_degree<1, 16> { static const int value = 6; };
template <> struct row_degree<1, 17> { static const int value = 6; };
template <> struct row_degree<1, 18> { static const int value = 6; };
template <> struct row_degree<1, 19> { static const int value = 6; };
template <> struct row_degree<1, 20> { static const int value = 6; };
template <> struct row_degree<1, 21> { static const int value = 6; };
template <> struct row_degree<1, 22> { static const int value = 5; };
template <> struct row_degree<1, 23> { static const int value = 5; };
template <> struct row_degree<1, 24> { static const int value = 6; };
template <> struct row_degree<1, 25> { static const int value = 5; };
template <> struct row_degree<1, 26> { static const int value = 5; };
template <> struct row_degree<1, 27> { static const int value = 4; };
template <> struct row_degree<1, 28> { static const int value = 5; };
template <> struct row_degree<1, 29> { static const int value = 5; };
template <> struct row_degree<1, 30> { static const int value = 5; };
template <> struct row_degree<1, 31> { static const int value = 5; };
template <> struct row_degree<1, 32> { static const int value = 5; };
template <> struct row_degree<1, 33> { static const int value = 5; };
template <> struct row_degree<1, 34> { static const int value = 5; };
template <> struct row_degree<1, 35> { static const int value = 5; };
template <> struct row_degree<1, 36> { static const int value = 5; };
template <> struct row_degree<1, 37> { static const int value = 4; };
template <> struct row_degree<1, 38> { static const int value = 5; };
template <> struct row_degree<1, 39> { static const int value = 5; };
template <> struct row_degree<1, 40> { static const int value = 4; };
template <> struct row_degree<1, 41> { static const int value = 5; };
template <> struct row_degree<1, 42> { static const int value = 4; };
template <> struct row_degree<1, 43> { static const int value = 5; };
template <> struct row_degree<1, 44> { static const int value = 5; };
template <> struct row_degree<1, 45> { static const int value = 4; };

template <> struct row_degree<2, 0> { static const int value = 8; };
template <> struct row_degree<2, 1> { static const int value = 10; };
template <> struct row_degree<2, 2> { static const int value = 8; };
template <> struct row_degree<2, 3> { static const int value = 10; };
template <> struct row_degree<2, 4> { static const int value = 4; };
template <> struct row_degree<2, 5> { static const int value = 6; };
template <> struct row_degree<2, 6> { static const int value = 6; };
template <> struct row_degree<2, 7> { static const int value = 6; };
template <> struct row_degree<2, 8> { static const int value = 4; };
template <> struct row_degree<2, 9> { static const int value = 5; };
template <> struct row_degree<2, 10> { static const int value = 5; };
template <> struct row_degree<2, 11> { static const int value = 5; };
template <> struct row_degree<2, 12> { static const int value = 4; };
template <> struct row_degree<2, 13> { static const int value = 5; };
template <> struct row_degree<2, 14> { static const int value = 5; };
template <> struct row_degree<2, 15> { static const int value = 4; };
template <> struct row_degree<2, 16> { static const int value = 5; };
template <> struct row_degree<2, 17> { static const int value = 5; };
template <> struct row_degree<2, 18> { static const int value = 4; };
template <> struct row_degree<2, 19> { static const int value = 4; };
template <> struct row_degree<2, 20> { static const int value = 4; };
template <> struct row_degree<2, 21> { static const int value = 4; };
template <> struct row_degree<2, 22> { static const int value = 3; };
template <> struct row_degree<2, 23> { static const int value = 4; };
template <> struct row_degree<2, 24> { static const int value = 4; };
template <> struct row_degree<2, 25> { static const int value = 3; };
template <> struct row_degree<2, 26> { static const int value = 5; };
template <> struct row_degree<2, 27> { static const int value = 3; };
template <> struct row_degree<2, 28> { static const int value = 4; };
template <> struct row_degree<2, 29> { static const int value = 3; };
template <> struct row_degree<2, 30> { static const int value = 5; };
template <> struct row_degree<2, 31> { static const int value = 3; };
template <> struct row_degree<2, 32> { static const int value = 4; };
template <> struct row_degree<2, 33> { static const int value = 4; };
template <> struct row_degree<2, 34> { static const int value = 4; };
template <> struct row_degree<2, 35> { static const int value = 4; };
template <> struct row_degree<2, 36> { static const int value = 4; };
template <> struct row_degree<2, 37> { static const int value = 3; };
template <> struct row_degree<2, 38> { static const int value = 4; };
template <> struct row_degree<2, 39> { static const int value = 4; };
template <> struct row_degree<2, 40> { static const int value = 4; };
template <> struct row_degree<2, 41> { static const int value = 4; };

////////////////////////////////////////////////////////////////////////
// nzs_row_degree
// Provides the number of nonzero shifted permutation matrices with non-zero shifts as
// a function of base graph and check node row. (Address calculations for zero shift
// values are simpler and can be more efficient.)
// See Tables 5.3.2-2 and 5.3.2-3, 3GPP 38.212
template <int BG, int ROW_INDEX> struct nzs_row_degree;
template <> struct nzs_row_degree<1, 0> { static const int value = 18; };
template <> struct nzs_row_degree<1, 1> { static const int value = 17; };
template <> struct nzs_row_degree<1, 2> { static const int value = 17; };
template <> struct nzs_row_degree<1, 3> { static const int value = 18; };
template <> struct nzs_row_degree<1, 4> { static const int value = 2; };
template <> struct nzs_row_degree<1, 5> { static const int value = 7; };
template <> struct nzs_row_degree<1, 6> { static const int value = 8; };
template <> struct nzs_row_degree<1, 7> { static const int value = 6; };
template <> struct nzs_row_degree<1, 8> { static const int value = 9; };
template <> struct nzs_row_degree<1, 9> { static const int value = 8; };
template <> struct nzs_row_degree<1, 10> { static const int value = 6; };
template <> struct nzs_row_degree<1, 11> { static const int value = 7; };
template <> struct nzs_row_degree<1, 12> { static const int value = 6; };
template <> struct nzs_row_degree<1, 13> { static const int value = 5; };
template <> struct nzs_row_degree<1, 14> { static const int value = 6; };
template <> struct nzs_row_degree<1, 15> { static const int value = 6; };
template <> struct nzs_row_degree<1, 16> { static const int value = 5; };
template <> struct nzs_row_degree<1, 17> { static const int value = 5; };
template <> struct nzs_row_degree<1, 18> { static const int value = 5; };
template <> struct nzs_row_degree<1, 19> { static const int value = 5; };
template <> struct nzs_row_degree<1, 20> { static const int value = 5; };
template <> struct nzs_row_degree<1, 21> { static const int value = 5; };
template <> struct nzs_row_degree<1, 22> { static const int value = 4; };
template <> struct nzs_row_degree<1, 23> { static const int value = 4; };
template <> struct nzs_row_degree<1, 24> { static const int value = 5; };
template <> struct nzs_row_degree<1, 25> { static const int value = 4; };
template <> struct nzs_row_degree<1, 26> { static const int value = 4; };
template <> struct nzs_row_degree<1, 27> { static const int value = 3; };
template <> struct nzs_row_degree<1, 28> { static const int value = 4; };
template <> struct nzs_row_degree<1, 29> { static const int value = 4; };
template <> struct nzs_row_degree<1, 30> { static const int value = 4; };
template <> struct nzs_row_degree<1, 31> { static const int value = 4; };
template <> struct nzs_row_degree<1, 32> { static const int value = 4; };
template <> struct nzs_row_degree<1, 33> { static const int value = 4; };
template <> struct nzs_row_degree<1, 34> { static const int value = 4; };
template <> struct nzs_row_degree<1, 35> { static const int value = 4; };
template <> struct nzs_row_degree<1, 36> { static const int value = 4; };
template <> struct nzs_row_degree<1, 37> { static const int value = 3; };
template <> struct nzs_row_degree<1, 38> { static const int value = 4; };
template <> struct nzs_row_degree<1, 39> { static const int value = 4; };
template <> struct nzs_row_degree<1, 40> { static const int value = 3; };
template <> struct nzs_row_degree<1, 41> { static const int value = 4; };
template <> struct nzs_row_degree<1, 42> { static const int value = 3; };
template <> struct nzs_row_degree<1, 43> { static const int value = 4; };
template <> struct nzs_row_degree<1, 44> { static const int value = 4; };
template <> struct nzs_row_degree<1, 45> { static const int value = 3; };

template <> struct nzs_row_degree<2, 0> { static const int value = 7; };
template <> struct nzs_row_degree<2, 1> { static const int value = 8; };
template <> struct nzs_row_degree<2, 2> { static const int value = 6; };
template <> struct nzs_row_degree<2, 3> { static const int value = 9; };
template <> struct nzs_row_degree<2, 4> { static const int value = 3; };
template <> struct nzs_row_degree<2, 5> { static const int value = 5; };
template <> struct nzs_row_degree<2, 6> { static const int value = 5; };
template <> struct nzs_row_degree<2, 7> { static const int value = 5; };
template <> struct nzs_row_degree<2, 8> { static const int value = 3; };
template <> struct nzs_row_degree<2, 9> { static const int value = 4; };
template <> struct nzs_row_degree<2, 10> { static const int value = 4; };
template <> struct nzs_row_degree<2, 11> { static const int value = 4; };
template <> struct nzs_row_degree<2, 12> { static const int value = 3; };
template <> struct nzs_row_degree<2, 13> { static const int value = 4; };
template <> struct nzs_row_degree<2, 14> { static const int value = 4; };
template <> struct nzs_row_degree<2, 15> { static const int value = 3; };
template <> struct nzs_row_degree<2, 16> { static const int value = 4; };
template <> struct nzs_row_degree<2, 17> { static const int value = 4; };
template <> struct nzs_row_degree<2, 18> { static const int value = 3; };
template <> struct nzs_row_degree<2, 19> { static const int value = 3; };
template <> struct nzs_row_degree<2, 20> { static const int value = 3; };
template <> struct nzs_row_degree<2, 21> { static const int value = 3; };
template <> struct nzs_row_degree<2, 22> { static const int value = 2; };
template <> struct nzs_row_degree<2, 23> { static const int value = 3; };
template <> struct nzs_row_degree<2, 24> { static const int value = 3; };
template <> struct nzs_row_degree<2, 25> { static const int value = 2; };
template <> struct nzs_row_degree<2, 26> { static const int value = 4; };
template <> struct nzs_row_degree<2, 27> { static const int value = 2; };
template <> struct nzs_row_degree<2, 28> { static const int value = 3; };
template <> struct nzs_row_degree<2, 29> { static const int value = 2; };
template <> struct nzs_row_degree<2, 30> { static const int value = 4; };
template <> struct nzs_row_degree<2, 31> { static const int value = 2; };
template <> struct nzs_row_degree<2, 32> { static const int value = 3; };
template <> struct nzs_row_degree<2, 33> { static const int value = 3; };
template <> struct nzs_row_degree<2, 34> { static const int value = 3; };
template <> struct nzs_row_degree<2, 35> { static const int value = 3; };
template <> struct nzs_row_degree<2, 36> { static const int value = 3; };
template <> struct nzs_row_degree<2, 37> { static const int value = 2; };
template <> struct nzs_row_degree<2, 38> { static const int value = 3; };
template <> struct nzs_row_degree<2, 39> { static const int value = 3; };
template <> struct nzs_row_degree<2, 40> { static const int value = 3; };
template <> struct nzs_row_degree<2, 41> { static const int value = 3; };

////////////////////////////////////////////////////////////////////////
// isolated_edge_count
// "Isolated" edges are base graph edges that are used by only 1 parity
// check node. As such, we do not need to update the stored values. (Use
// during the next iteration would subtract the same value that was
// added.)
template <bool THasIsolatedEdge> struct isolated_edge_count;
template <> struct isolated_edge_count<true>  { static const int value = 1; };
template <> struct isolated_edge_count<false> { static const int value = 0; };

////////////////////////////////////////////////////////////////////////
// update_row_degree
// Parity check nodes with an isolated edge at the end can avoid updating
// the value, since it is only used by one node. For 5G base graphs, rows 4
// and higher have an isolated edge.
template <int BG, int ROW_INDEX>
struct update_row_degree
{
    static const int value = (row_degree<BG, ROW_INDEX>::value - isolated_edge_count<(ROW_INDEX > 3)>::value);
};

////////////////////////////////////////////////////////////////////////
// vnode_index
// Variable node indices (shared by all lifting sets for a given base
// graph).
// See Tables 5.3.2-2 and 5.3.2-3, 3GPP 38.212
template <int BG, int CHECK_NODE, int INDEX> struct vnode_index;
// BG1 CHECK NODE 0 (degree: 19)
template <> struct vnode_index<1, 0, 0> { static const int value = 0; };
template <> struct vnode_index<1, 0, 1> { static const int value = 1; };
template <> struct vnode_index<1, 0, 2> { static const int value = 2; };
template <> struct vnode_index<1, 0, 3> { static const int value = 3; };
template <> struct vnode_index<1, 0, 4> { static const int value = 5; };
template <> struct vnode_index<1, 0, 5> { static const int value = 6; };
template <> struct vnode_index<1, 0, 6> { static const int value = 9; };
template <> struct vnode_index<1, 0, 7> { static const int value = 10; };
template <> struct vnode_index<1, 0, 8> { static const int value = 11; };
template <> struct vnode_index<1, 0, 9> { static const int value = 12; };
template <> struct vnode_index<1, 0, 10> { static const int value = 13; };
template <> struct vnode_index<1, 0, 11> { static const int value = 15; };
template <> struct vnode_index<1, 0, 12> { static const int value = 16; };
template <> struct vnode_index<1, 0, 13> { static const int value = 18; };
template <> struct vnode_index<1, 0, 14> { static const int value = 19; };
template <> struct vnode_index<1, 0, 15> { static const int value = 20; };
template <> struct vnode_index<1, 0, 16> { static const int value = 21; };
template <> struct vnode_index<1, 0, 17> { static const int value = 22; };
template <> struct vnode_index<1, 0, 18> { static const int value = 23; };
// BG1 CHECK NODE 1 (degree: 19)
template <> struct vnode_index<1, 1, 0> { static const int value = 0; };
template <> struct vnode_index<1, 1, 1> { static const int value = 2; };
template <> struct vnode_index<1, 1, 2> { static const int value = 3; };
template <> struct vnode_index<1, 1, 3> { static const int value = 4; };
template <> struct vnode_index<1, 1, 4> { static const int value = 5; };
template <> struct vnode_index<1, 1, 5> { static const int value = 7; };
template <> struct vnode_index<1, 1, 6> { static const int value = 8; };
template <> struct vnode_index<1, 1, 7> { static const int value = 9; };
template <> struct vnode_index<1, 1, 8> { static const int value = 11; };
template <> struct vnode_index<1, 1, 9> { static const int value = 12; };
template <> struct vnode_index<1, 1, 10> { static const int value = 14; };
template <> struct vnode_index<1, 1, 11> { static const int value = 15; };
template <> struct vnode_index<1, 1, 12> { static const int value = 16; };
template <> struct vnode_index<1, 1, 13> { static const int value = 17; };
template <> struct vnode_index<1, 1, 14> { static const int value = 19; };
template <> struct vnode_index<1, 1, 15> { static const int value = 21; };
template <> struct vnode_index<1, 1, 16> { static const int value = 22; };
template <> struct vnode_index<1, 1, 17> { static const int value = 23; };
template <> struct vnode_index<1, 1, 18> { static const int value = 24; };
// BG1 CHECK NODE 2 (degree: 19)
template <> struct vnode_index<1, 2, 0> { static const int value = 0; };
template <> struct vnode_index<1, 2, 1> { static const int value = 1; };
template <> struct vnode_index<1, 2, 2> { static const int value = 2; };
template <> struct vnode_index<1, 2, 3> { static const int value = 4; };
template <> struct vnode_index<1, 2, 4> { static const int value = 5; };
template <> struct vnode_index<1, 2, 5> { static const int value = 6; };
template <> struct vnode_index<1, 2, 6> { static const int value = 7; };
template <> struct vnode_index<1, 2, 7> { static const int value = 8; };
template <> struct vnode_index<1, 2, 8> { static const int value = 9; };
template <> struct vnode_index<1, 2, 9> { static const int value = 10; };
template <> struct vnode_index<1, 2, 10> { static const int value = 13; };
template <> struct vnode_index<1, 2, 11> { static const int value = 14; };
template <> struct vnode_index<1, 2, 12> { static const int value = 15; };
template <> struct vnode_index<1, 2, 13> { static const int value = 17; };
template <> struct vnode_index<1, 2, 14> { static const int value = 18; };
template <> struct vnode_index<1, 2, 15> { static const int value = 19; };
template <> struct vnode_index<1, 2, 16> { static const int value = 20; };
template <> struct vnode_index<1, 2, 17> { static const int value = 24; };
template <> struct vnode_index<1, 2, 18> { static const int value = 25; };
// BG1 CHECK NODE 3 (degree: 19)
template <> struct vnode_index<1, 3, 0> { static const int value = 0; };
template <> struct vnode_index<1, 3, 1> { static const int value = 1; };
template <> struct vnode_index<1, 3, 2> { static const int value = 3; };
template <> struct vnode_index<1, 3, 3> { static const int value = 4; };
template <> struct vnode_index<1, 3, 4> { static const int value = 6; };
template <> struct vnode_index<1, 3, 5> { static const int value = 7; };
template <> struct vnode_index<1, 3, 6> { static const int value = 8; };
template <> struct vnode_index<1, 3, 7> { static const int value = 10; };
template <> struct vnode_index<1, 3, 8> { static const int value = 11; };
template <> struct vnode_index<1, 3, 9> { static const int value = 12; };
template <> struct vnode_index<1, 3, 10> { static const int value = 13; };
template <> struct vnode_index<1, 3, 11> { static const int value = 14; };
template <> struct vnode_index<1, 3, 12> { static const int value = 16; };
template <> struct vnode_index<1, 3, 13> { static const int value = 17; };
template <> struct vnode_index<1, 3, 14> { static const int value = 18; };
template <> struct vnode_index<1, 3, 15> { static const int value = 20; };
template <> struct vnode_index<1, 3, 16> { static const int value = 21; };
template <> struct vnode_index<1, 3, 17> { static const int value = 22; };
template <> struct vnode_index<1, 3, 18> { static const int value = 25; };
// BG1 CHECK NODE 4 (degree: 3)
template <> struct vnode_index<1, 4, 0> { static const int value = 0; };
template <> struct vnode_index<1, 4, 1> { static const int value = 1; };
template <> struct vnode_index<1, 4, 2> { static const int value = 26; };
// BG1 CHECK NODE 5 (degree: 8)
template <> struct vnode_index<1, 5, 0> { static const int value = 0; };
template <> struct vnode_index<1, 5, 1> { static const int value = 1; };
template <> struct vnode_index<1, 5, 2> { static const int value = 3; };
template <> struct vnode_index<1, 5, 3> { static const int value = 12; };
template <> struct vnode_index<1, 5, 4> { static const int value = 16; };
template <> struct vnode_index<1, 5, 5> { static const int value = 21; };
template <> struct vnode_index<1, 5, 6> { static const int value = 22; };
template <> struct vnode_index<1, 5, 7> { static const int value = 27; };
// BG1 CHECK NODE 6 (degree: 9)
template <> struct vnode_index<1, 6, 0> { static const int value = 0; };
template <> struct vnode_index<1, 6, 1> { static const int value = 6; };
template <> struct vnode_index<1, 6, 2> { static const int value = 10; };
template <> struct vnode_index<1, 6, 3> { static const int value = 11; };
template <> struct vnode_index<1, 6, 4> { static const int value = 13; };
template <> struct vnode_index<1, 6, 5> { static const int value = 17; };
template <> struct vnode_index<1, 6, 6> { static const int value = 18; };
template <> struct vnode_index<1, 6, 7> { static const int value = 20; };
template <> struct vnode_index<1, 6, 8> { static const int value = 28; };
// BG1 CHECK NODE 7 (degree: 7)
template <> struct vnode_index<1, 7, 0> { static const int value = 0; };
template <> struct vnode_index<1, 7, 1> { static const int value = 1; };
template <> struct vnode_index<1, 7, 2> { static const int value = 4; };
template <> struct vnode_index<1, 7, 3> { static const int value = 7; };
template <> struct vnode_index<1, 7, 4> { static const int value = 8; };
template <> struct vnode_index<1, 7, 5> { static const int value = 14; };
template <> struct vnode_index<1, 7, 6> { static const int value = 29; };
// BG1 CHECK NODE 8 (degree: 10)
template <> struct vnode_index<1, 8, 0> { static const int value = 0; };
template <> struct vnode_index<1, 8, 1> { static const int value = 1; };
template <> struct vnode_index<1, 8, 2> { static const int value = 3; };
template <> struct vnode_index<1, 8, 3> { static const int value = 12; };
template <> struct vnode_index<1, 8, 4> { static const int value = 16; };
template <> struct vnode_index<1, 8, 5> { static const int value = 19; };
template <> struct vnode_index<1, 8, 6> { static const int value = 21; };
template <> struct vnode_index<1, 8, 7> { static const int value = 22; };
template <> struct vnode_index<1, 8, 8> { static const int value = 24; };
template <> struct vnode_index<1, 8, 9> { static const int value = 30; };
// BG1 CHECK NODE 9 (degree: 9)
template <> struct vnode_index<1, 9, 0> { static const int value = 0; };
template <> struct vnode_index<1, 9, 1> { static const int value = 1; };
template <> struct vnode_index<1, 9, 2> { static const int value = 10; };
template <> struct vnode_index<1, 9, 3> { static const int value = 11; };
template <> struct vnode_index<1, 9, 4> { static const int value = 13; };
template <> struct vnode_index<1, 9, 5> { static const int value = 17; };
template <> struct vnode_index<1, 9, 6> { static const int value = 18; };
template <> struct vnode_index<1, 9, 7> { static const int value = 20; };
template <> struct vnode_index<1, 9, 8> { static const int value = 31; };
// BG1 CHECK NODE 10 (degree: 7)
template <> struct vnode_index<1, 10, 0> { static const int value = 1; };
template <> struct vnode_index<1, 10, 1> { static const int value = 2; };
template <> struct vnode_index<1, 10, 2> { static const int value = 4; };
template <> struct vnode_index<1, 10, 3> { static const int value = 7; };
template <> struct vnode_index<1, 10, 4> { static const int value = 8; };
template <> struct vnode_index<1, 10, 5> { static const int value = 14; };
template <> struct vnode_index<1, 10, 6> { static const int value = 32; };
// BG1 CHECK NODE 11 (degree: 8)
template <> struct vnode_index<1, 11, 0> { static const int value = 0; };
template <> struct vnode_index<1, 11, 1> { static const int value = 1; };
template <> struct vnode_index<1, 11, 2> { static const int value = 12; };
template <> struct vnode_index<1, 11, 3> { static const int value = 16; };
template <> struct vnode_index<1, 11, 4> { static const int value = 21; };
template <> struct vnode_index<1, 11, 5> { static const int value = 22; };
template <> struct vnode_index<1, 11, 6> { static const int value = 23; };
template <> struct vnode_index<1, 11, 7> { static const int value = 33; };
// BG1 CHECK NODE 12 (degree: 7)
template <> struct vnode_index<1, 12, 0> { static const int value = 0; };
template <> struct vnode_index<1, 12, 1> { static const int value = 1; };
template <> struct vnode_index<1, 12, 2> { static const int value = 10; };
template <> struct vnode_index<1, 12, 3> { static const int value = 11; };
template <> struct vnode_index<1, 12, 4> { static const int value = 13; };
template <> struct vnode_index<1, 12, 5> { static const int value = 18; };
template <> struct vnode_index<1, 12, 6> { static const int value = 34; };
// BG1 CHECK NODE 13 (degree: 6)
template <> struct vnode_index<1, 13, 0> { static const int value = 0; };
template <> struct vnode_index<1, 13, 1> { static const int value = 3; };
template <> struct vnode_index<1, 13, 2> { static const int value = 7; };
template <> struct vnode_index<1, 13, 3> { static const int value = 20; };
template <> struct vnode_index<1, 13, 4> { static const int value = 23; };
template <> struct vnode_index<1, 13, 5> { static const int value = 35; };
// BG1 CHECK NODE 14 (degree: 7)
template <> struct vnode_index<1, 14, 0> { static const int value = 0; };
template <> struct vnode_index<1, 14, 1> { static const int value = 12; };
template <> struct vnode_index<1, 14, 2> { static const int value = 15; };
template <> struct vnode_index<1, 14, 3> { static const int value = 16; };
template <> struct vnode_index<1, 14, 4> { static const int value = 17; };
template <> struct vnode_index<1, 14, 5> { static const int value = 21; };
template <> struct vnode_index<1, 14, 6> { static const int value = 36; };
// BG1 CHECK NODE 15 (degree: 7)
template <> struct vnode_index<1, 15, 0> { static const int value = 0; };
template <> struct vnode_index<1, 15, 1> { static const int value = 1; };
template <> struct vnode_index<1, 15, 2> { static const int value = 10; };
template <> struct vnode_index<1, 15, 3> { static const int value = 13; };
template <> struct vnode_index<1, 15, 4> { static const int value = 18; };
template <> struct vnode_index<1, 15, 5> { static const int value = 25; };
template <> struct vnode_index<1, 15, 6> { static const int value = 37; };
// BG1 CHECK NODE 16 (degree: 6)
template <> struct vnode_index<1, 16, 0> { static const int value = 1; };
template <> struct vnode_index<1, 16, 1> { static const int value = 3; };
template <> struct vnode_index<1, 16, 2> { static const int value = 11; };
template <> struct vnode_index<1, 16, 3> { static const int value = 20; };
template <> struct vnode_index<1, 16, 4> { static const int value = 22; };
template <> struct vnode_index<1, 16, 5> { static const int value = 38; };
// BG1 CHECK NODE 17 (degree: 6)
template <> struct vnode_index<1, 17, 0> { static const int value = 0; };
template <> struct vnode_index<1, 17, 1> { static const int value = 14; };
template <> struct vnode_index<1, 17, 2> { static const int value = 16; };
template <> struct vnode_index<1, 17, 3> { static const int value = 17; };
template <> struct vnode_index<1, 17, 4> { static const int value = 21; };
template <> struct vnode_index<1, 17, 5> { static const int value = 39; };
// BG1 CHECK NODE 18 (degree: 6)
template <> struct vnode_index<1, 18, 0> { static const int value = 1; };
template <> struct vnode_index<1, 18, 1> { static const int value = 12; };
template <> struct vnode_index<1, 18, 2> { static const int value = 13; };
template <> struct vnode_index<1, 18, 3> { static const int value = 18; };
template <> struct vnode_index<1, 18, 4> { static const int value = 19; };
template <> struct vnode_index<1, 18, 5> { static const int value = 40; };
// BG1 CHECK NODE 19 (degree: 6)
template <> struct vnode_index<1, 19, 0> { static const int value = 0; };
template <> struct vnode_index<1, 19, 1> { static const int value = 1; };
template <> struct vnode_index<1, 19, 2> { static const int value = 7; };
template <> struct vnode_index<1, 19, 3> { static const int value = 8; };
template <> struct vnode_index<1, 19, 4> { static const int value = 10; };
template <> struct vnode_index<1, 19, 5> { static const int value = 41; };
// BG1 CHECK NODE 20 (degree: 6)
template <> struct vnode_index<1, 20, 0> { static const int value = 0; };
template <> struct vnode_index<1, 20, 1> { static const int value = 3; };
template <> struct vnode_index<1, 20, 2> { static const int value = 9; };
template <> struct vnode_index<1, 20, 3> { static const int value = 11; };
template <> struct vnode_index<1, 20, 4> { static const int value = 22; };
template <> struct vnode_index<1, 20, 5> { static const int value = 42; };
// BG1 CHECK NODE 21 (degree: 6)
template <> struct vnode_index<1, 21, 0> { static const int value = 1; };
template <> struct vnode_index<1, 21, 1> { static const int value = 5; };
template <> struct vnode_index<1, 21, 2> { static const int value = 16; };
template <> struct vnode_index<1, 21, 3> { static const int value = 20; };
template <> struct vnode_index<1, 21, 4> { static const int value = 21; };
template <> struct vnode_index<1, 21, 5> { static const int value = 43; };
// BG1 CHECK NODE 22 (degree: 5)
template <> struct vnode_index<1, 22, 0> { static const int value = 0; };
template <> struct vnode_index<1, 22, 1> { static const int value = 12; };
template <> struct vnode_index<1, 22, 2> { static const int value = 13; };
template <> struct vnode_index<1, 22, 3> { static const int value = 17; };
template <> struct vnode_index<1, 22, 4> { static const int value = 44; };
// BG1 CHECK NODE 23 (degree: 5)
template <> struct vnode_index<1, 23, 0> { static const int value = 1; };
template <> struct vnode_index<1, 23, 1> { static const int value = 2; };
template <> struct vnode_index<1, 23, 2> { static const int value = 10; };
template <> struct vnode_index<1, 23, 3> { static const int value = 18; };
template <> struct vnode_index<1, 23, 4> { static const int value = 45; };
// BG1 CHECK NODE 24 (degree: 6)
template <> struct vnode_index<1, 24, 0> { static const int value = 0; };
template <> struct vnode_index<1, 24, 1> { static const int value = 3; };
template <> struct vnode_index<1, 24, 2> { static const int value = 4; };
template <> struct vnode_index<1, 24, 3> { static const int value = 11; };
template <> struct vnode_index<1, 24, 4> { static const int value = 22; };
template <> struct vnode_index<1, 24, 5> { static const int value = 46; };
// BG1 CHECK NODE 25 (degree: 5)
template <> struct vnode_index<1, 25, 0> { static const int value = 1; };
template <> struct vnode_index<1, 25, 1> { static const int value = 6; };
template <> struct vnode_index<1, 25, 2> { static const int value = 7; };
template <> struct vnode_index<1, 25, 3> { static const int value = 14; };
template <> struct vnode_index<1, 25, 4> { static const int value = 47; };
// BG1 CHECK NODE 26 (degree: 5)
template <> struct vnode_index<1, 26, 0> { static const int value = 0; };
template <> struct vnode_index<1, 26, 1> { static const int value = 2; };
template <> struct vnode_index<1, 26, 2> { static const int value = 4; };
template <> struct vnode_index<1, 26, 3> { static const int value = 15; };
template <> struct vnode_index<1, 26, 4> { static const int value = 48; };
// BG1 CHECK NODE 27 (degree: 4)
template <> struct vnode_index<1, 27, 0> { static const int value = 1; };
template <> struct vnode_index<1, 27, 1> { static const int value = 6; };
template <> struct vnode_index<1, 27, 2> { static const int value = 8; };
template <> struct vnode_index<1, 27, 3> { static const int value = 49; };
// BG1 CHECK NODE 28 (degree: 5)
template <> struct vnode_index<1, 28, 0> { static const int value = 0; };
template <> struct vnode_index<1, 28, 1> { static const int value = 4; };
template <> struct vnode_index<1, 28, 2> { static const int value = 19; };
template <> struct vnode_index<1, 28, 3> { static const int value = 21; };
template <> struct vnode_index<1, 28, 4> { static const int value = 50; };
// BG1 CHECK NODE 29 (degree: 5)
template <> struct vnode_index<1, 29, 0> { static const int value = 1; };
template <> struct vnode_index<1, 29, 1> { static const int value = 14; };
template <> struct vnode_index<1, 29, 2> { static const int value = 18; };
template <> struct vnode_index<1, 29, 3> { static const int value = 25; };
template <> struct vnode_index<1, 29, 4> { static const int value = 51; };
// BG1 CHECK NODE 30 (degree: 5)
template <> struct vnode_index<1, 30, 0> { static const int value = 0; };
template <> struct vnode_index<1, 30, 1> { static const int value = 10; };
template <> struct vnode_index<1, 30, 2> { static const int value = 13; };
template <> struct vnode_index<1, 30, 3> { static const int value = 24; };
template <> struct vnode_index<1, 30, 4> { static const int value = 52; };
// BG1 CHECK NODE 31 (degree: 5)
template <> struct vnode_index<1, 31, 0> { static const int value = 1; };
template <> struct vnode_index<1, 31, 1> { static const int value = 7; };
template <> struct vnode_index<1, 31, 2> { static const int value = 22; };
template <> struct vnode_index<1, 31, 3> { static const int value = 25; };
template <> struct vnode_index<1, 31, 4> { static const int value = 53; };
// BG1 CHECK NODE 32 (degree: 5)
template <> struct vnode_index<1, 32, 0> { static const int value = 0; };
template <> struct vnode_index<1, 32, 1> { static const int value = 12; };
template <> struct vnode_index<1, 32, 2> { static const int value = 14; };
template <> struct vnode_index<1, 32, 3> { static const int value = 24; };
template <> struct vnode_index<1, 32, 4> { static const int value = 54; };
// BG1 CHECK NODE 33 (degree: 5)
template <> struct vnode_index<1, 33, 0> { static const int value = 1; };
template <> struct vnode_index<1, 33, 1> { static const int value = 2; };
template <> struct vnode_index<1, 33, 2> { static const int value = 11; };
template <> struct vnode_index<1, 33, 3> { static const int value = 21; };
template <> struct vnode_index<1, 33, 4> { static const int value = 55; };
// BG1 CHECK NODE 34 (degree: 5)
template <> struct vnode_index<1, 34, 0> { static const int value = 0; };
template <> struct vnode_index<1, 34, 1> { static const int value = 7; };
template <> struct vnode_index<1, 34, 2> { static const int value = 15; };
template <> struct vnode_index<1, 34, 3> { static const int value = 17; };
template <> struct vnode_index<1, 34, 4> { static const int value = 56; };
// BG1 CHECK NODE 35 (degree: 5)
template <> struct vnode_index<1, 35, 0> { static const int value = 1; };
template <> struct vnode_index<1, 35, 1> { static const int value = 6; };
template <> struct vnode_index<1, 35, 2> { static const int value = 12; };
template <> struct vnode_index<1, 35, 3> { static const int value = 22; };
template <> struct vnode_index<1, 35, 4> { static const int value = 57; };
// BG1 CHECK NODE 36 (degree: 5)
template <> struct vnode_index<1, 36, 0> { static const int value = 0; };
template <> struct vnode_index<1, 36, 1> { static const int value = 14; };
template <> struct vnode_index<1, 36, 2> { static const int value = 15; };
template <> struct vnode_index<1, 36, 3> { static const int value = 18; };
template <> struct vnode_index<1, 36, 4> { static const int value = 58; };
// BG1 CHECK NODE 37 (degree: 4)
template <> struct vnode_index<1, 37, 0> { static const int value = 1; };
template <> struct vnode_index<1, 37, 1> { static const int value = 13; };
template <> struct vnode_index<1, 37, 2> { static const int value = 23; };
template <> struct vnode_index<1, 37, 3> { static const int value = 59; };
// BG1 CHECK NODE 38 (degree: 5)
template <> struct vnode_index<1, 38, 0> { static const int value = 0; };
template <> struct vnode_index<1, 38, 1> { static const int value = 9; };
template <> struct vnode_index<1, 38, 2> { static const int value = 10; };
template <> struct vnode_index<1, 38, 3> { static const int value = 12; };
template <> struct vnode_index<1, 38, 4> { static const int value = 60; };
// BG1 CHECK NODE 39 (degree: 5)
template <> struct vnode_index<1, 39, 0> { static const int value = 1; };
template <> struct vnode_index<1, 39, 1> { static const int value = 3; };
template <> struct vnode_index<1, 39, 2> { static const int value = 7; };
template <> struct vnode_index<1, 39, 3> { static const int value = 19; };
template <> struct vnode_index<1, 39, 4> { static const int value = 61; };
// BG1 CHECK NODE 40 (degree: 4)
template <> struct vnode_index<1, 40, 0> { static const int value = 0; };
template <> struct vnode_index<1, 40, 1> { static const int value = 8; };
template <> struct vnode_index<1, 40, 2> { static const int value = 17; };
template <> struct vnode_index<1, 40, 3> { static const int value = 62; };
// BG1 CHECK NODE 41 (degree: 5)
template <> struct vnode_index<1, 41, 0> { static const int value = 1; };
template <> struct vnode_index<1, 41, 1> { static const int value = 3; };
template <> struct vnode_index<1, 41, 2> { static const int value = 9; };
template <> struct vnode_index<1, 41, 3> { static const int value = 18; };
template <> struct vnode_index<1, 41, 4> { static const int value = 63; };
// BG1 CHECK NODE 42 (degree: 4)
template <> struct vnode_index<1, 42, 0> { static const int value = 0; };
template <> struct vnode_index<1, 42, 1> { static const int value = 4; };
template <> struct vnode_index<1, 42, 2> { static const int value = 24; };
template <> struct vnode_index<1, 42, 3> { static const int value = 64; };
// BG1 CHECK NODE 43 (degree: 5)
template <> struct vnode_index<1, 43, 0> { static const int value = 1; };
template <> struct vnode_index<1, 43, 1> { static const int value = 16; };
template <> struct vnode_index<1, 43, 2> { static const int value = 18; };
template <> struct vnode_index<1, 43, 3> { static const int value = 25; };
template <> struct vnode_index<1, 43, 4> { static const int value = 65; };
// BG1 CHECK NODE 44 (degree: 5)
template <> struct vnode_index<1, 44, 0> { static const int value = 0; };
template <> struct vnode_index<1, 44, 1> { static const int value = 7; };
template <> struct vnode_index<1, 44, 2> { static const int value = 9; };
template <> struct vnode_index<1, 44, 3> { static const int value = 22; };
template <> struct vnode_index<1, 44, 4> { static const int value = 66; };
// BG1 CHECK NODE 45 (degree: 4)
template <> struct vnode_index<1, 45, 0> { static const int value = 1; };
template <> struct vnode_index<1, 45, 1> { static const int value = 6; };
template <> struct vnode_index<1, 45, 2> { static const int value = 10; };
template <> struct vnode_index<1, 45, 3> { static const int value = 67; };

// BG2 CHECK NODE 0 (degree: 8)
template <> struct vnode_index<2, 0, 0> { static const int value = 0; };
template <> struct vnode_index<2, 0, 1> { static const int value = 1; };
template <> struct vnode_index<2, 0, 2> { static const int value = 2; };
template <> struct vnode_index<2, 0, 3> { static const int value = 3; };
template <> struct vnode_index<2, 0, 4> { static const int value = 6; };
template <> struct vnode_index<2, 0, 5> { static const int value = 9; };
template <> struct vnode_index<2, 0, 6> { static const int value = 10; };
template <> struct vnode_index<2, 0, 7> { static const int value = 11; };
// BG2 CHECK NODE 1 (degree: 10)
template <> struct vnode_index<2, 1, 0> { static const int value = 0; };
template <> struct vnode_index<2, 1, 1> { static const int value = 3; };
template <> struct vnode_index<2, 1, 2> { static const int value = 4; };
template <> struct vnode_index<2, 1, 3> { static const int value = 5; };
template <> struct vnode_index<2, 1, 4> { static const int value = 6; };
template <> struct vnode_index<2, 1, 5> { static const int value = 7; };
template <> struct vnode_index<2, 1, 6> { static const int value = 8; };
template <> struct vnode_index<2, 1, 7> { static const int value = 9; };
template <> struct vnode_index<2, 1, 8> { static const int value = 11; };
template <> struct vnode_index<2, 1, 9> { static const int value = 12; };
// BG2 CHECK NODE 2 (degree: 8)
template <> struct vnode_index<2, 2, 0> { static const int value = 0; };
template <> struct vnode_index<2, 2, 1> { static const int value = 1; };
template <> struct vnode_index<2, 2, 2> { static const int value = 3; };
template <> struct vnode_index<2, 2, 3> { static const int value = 4; };
template <> struct vnode_index<2, 2, 4> { static const int value = 8; };
template <> struct vnode_index<2, 2, 5> { static const int value = 10; };
template <> struct vnode_index<2, 2, 6> { static const int value = 12; };
template <> struct vnode_index<2, 2, 7> { static const int value = 13; };
// BG2 CHECK NODE 3 (degree: 10)
template <> struct vnode_index<2, 3, 0> { static const int value = 1; };
template <> struct vnode_index<2, 3, 1> { static const int value = 2; };
template <> struct vnode_index<2, 3, 2> { static const int value = 4; };
template <> struct vnode_index<2, 3, 3> { static const int value = 5; };
template <> struct vnode_index<2, 3, 4> { static const int value = 6; };
template <> struct vnode_index<2, 3, 5> { static const int value = 7; };
template <> struct vnode_index<2, 3, 6> { static const int value = 8; };
template <> struct vnode_index<2, 3, 7> { static const int value = 9; };
template <> struct vnode_index<2, 3, 8> { static const int value = 10; };
template <> struct vnode_index<2, 3, 9> { static const int value = 13; };
// BG2 CHECK NODE 4 (degree: 4)
template <> struct vnode_index<2, 4, 0> { static const int value = 0; };
template <> struct vnode_index<2, 4, 1> { static const int value = 1; };
template <> struct vnode_index<2, 4, 2> { static const int value = 11; };
template <> struct vnode_index<2, 4, 3> { static const int value = 14; };
// BG2 CHECK NODE 5 (degree: 6)
template <> struct vnode_index<2, 5, 0> { static const int value = 0; };
template <> struct vnode_index<2, 5, 1> { static const int value = 1; };
template <> struct vnode_index<2, 5, 2> { static const int value = 5; };
template <> struct vnode_index<2, 5, 3> { static const int value = 7; };
template <> struct vnode_index<2, 5, 4> { static const int value = 11; };
template <> struct vnode_index<2, 5, 5> { static const int value = 15; };
// BG2 CHECK NODE 6 (degree: 6)
template <> struct vnode_index<2, 6, 0> { static const int value = 0; };
template <> struct vnode_index<2, 6, 1> { static const int value = 5; };
template <> struct vnode_index<2, 6, 2> { static const int value = 7; };
template <> struct vnode_index<2, 6, 3> { static const int value = 9; };
template <> struct vnode_index<2, 6, 4> { static const int value = 11; };
template <> struct vnode_index<2, 6, 5> { static const int value = 16; };
// BG2 CHECK NODE 7 (degree: 6)
template <> struct vnode_index<2, 7, 0> { static const int value = 1; };
template <> struct vnode_index<2, 7, 1> { static const int value = 5; };
template <> struct vnode_index<2, 7, 2> { static const int value = 7; };
template <> struct vnode_index<2, 7, 3> { static const int value = 11; };
template <> struct vnode_index<2, 7, 4> { static const int value = 13; };
template <> struct vnode_index<2, 7, 5> { static const int value = 17; };
// BG2 CHECK NODE 8 (degree: 4)
template <> struct vnode_index<2, 8, 0> { static const int value = 0; };
template <> struct vnode_index<2, 8, 1> { static const int value = 1; };
template <> struct vnode_index<2, 8, 2> { static const int value = 12; };
template <> struct vnode_index<2, 8, 3> { static const int value = 18; };
// BG2 CHECK NODE 9 (degree: 5)
template <> struct vnode_index<2, 9, 0> { static const int value = 1; };
template <> struct vnode_index<2, 9, 1> { static const int value = 8; };
template <> struct vnode_index<2, 9, 2> { static const int value = 10; };
template <> struct vnode_index<2, 9, 3> { static const int value = 11; };
template <> struct vnode_index<2, 9, 4> { static const int value = 19; };
// BG2 CHECK NODE 10 (degree: 5)
template <> struct vnode_index<2, 10, 0> { static const int value = 0; };
template <> struct vnode_index<2, 10, 1> { static const int value = 1; };
template <> struct vnode_index<2, 10, 2> { static const int value = 6; };
template <> struct vnode_index<2, 10, 3> { static const int value = 7; };
template <> struct vnode_index<2, 10, 4> { static const int value = 20; };
// BG2 CHECK NODE 11 (degree: 5)
template <> struct vnode_index<2, 11, 0> { static const int value = 0; };
template <> struct vnode_index<2, 11, 1> { static const int value = 7; };
template <> struct vnode_index<2, 11, 2> { static const int value = 9; };
template <> struct vnode_index<2, 11, 3> { static const int value = 13; };
template <> struct vnode_index<2, 11, 4> { static const int value = 21; };
// BG2 CHECK NODE 12 (degree: 4)
template <> struct vnode_index<2, 12, 0> { static const int value = 1; };
template <> struct vnode_index<2, 12, 1> { static const int value = 3; };
template <> struct vnode_index<2, 12, 2> { static const int value = 11; };
template <> struct vnode_index<2, 12, 3> { static const int value = 22; };
// BG2 CHECK NODE 13 (degree: 5)
template <> struct vnode_index<2, 13, 0> { static const int value = 0; };
template <> struct vnode_index<2, 13, 1> { static const int value = 1; };
template <> struct vnode_index<2, 13, 2> { static const int value = 8; };
template <> struct vnode_index<2, 13, 3> { static const int value = 13; };
template <> struct vnode_index<2, 13, 4> { static const int value = 23; };
// BG2 CHECK NODE 14 (degree: 5)
template <> struct vnode_index<2, 14, 0> { static const int value = 1; };
template <> struct vnode_index<2, 14, 1> { static const int value = 6; };
template <> struct vnode_index<2, 14, 2> { static const int value = 11; };
template <> struct vnode_index<2, 14, 3> { static const int value = 13; };
template <> struct vnode_index<2, 14, 4> { static const int value = 24; };
// BG2 CHECK NODE 15 (degree: 4)
template <> struct vnode_index<2, 15, 0> { static const int value = 0; };
template <> struct vnode_index<2, 15, 1> { static const int value = 10; };
template <> struct vnode_index<2, 15, 2> { static const int value = 11; };
template <> struct vnode_index<2, 15, 3> { static const int value = 25; };
// BG2 CHECK NODE 16 (degree: 5)
template <> struct vnode_index<2, 16, 0> { static const int value = 1; };
template <> struct vnode_index<2, 16, 1> { static const int value = 9; };
template <> struct vnode_index<2, 16, 2> { static const int value = 11; };
template <> struct vnode_index<2, 16, 3> { static const int value = 12; };
template <> struct vnode_index<2, 16, 4> { static const int value = 26; };
// BG2 CHECK NODE 17 (degree: 5)
template <> struct vnode_index<2, 17, 0> { static const int value = 1; };
template <> struct vnode_index<2, 17, 1> { static const int value = 5; };
template <> struct vnode_index<2, 17, 2> { static const int value = 11; };
template <> struct vnode_index<2, 17, 3> { static const int value = 12; };
template <> struct vnode_index<2, 17, 4> { static const int value = 27; };
// BG2 CHECK NODE 18 (degree: 4)
template <> struct vnode_index<2, 18, 0> { static const int value = 0; };
template <> struct vnode_index<2, 18, 1> { static const int value = 6; };
template <> struct vnode_index<2, 18, 2> { static const int value = 7; };
template <> struct vnode_index<2, 18, 3> { static const int value = 28; };
// BG2 CHECK NODE 19 (degree: 4)
template <> struct vnode_index<2, 19, 0> { static const int value = 0; };
template <> struct vnode_index<2, 19, 1> { static const int value = 1; };
template <> struct vnode_index<2, 19, 2> { static const int value = 10; };
template <> struct vnode_index<2, 19, 3> { static const int value = 29; };
// BG2 CHECK NODE 20 (degree: 4)
template <> struct vnode_index<2, 20, 0> { static const int value = 1; };
template <> struct vnode_index<2, 20, 1> { static const int value = 4; };
template <> struct vnode_index<2, 20, 2> { static const int value = 11; };
template <> struct vnode_index<2, 20, 3> { static const int value = 30; };
// BG2 CHECK NODE 21 (degree: 4)
template <> struct vnode_index<2, 21, 0> { static const int value = 0; };
template <> struct vnode_index<2, 21, 1> { static const int value = 8; };
template <> struct vnode_index<2, 21, 2> { static const int value = 13; };
template <> struct vnode_index<2, 21, 3> { static const int value = 31; };
// BG2 CHECK NODE 22 (degree: 3)
template <> struct vnode_index<2, 22, 0> { static const int value = 1; };
template <> struct vnode_index<2, 22, 1> { static const int value = 2; };
template <> struct vnode_index<2, 22, 2> { static const int value = 32; };
// BG2 CHECK NODE 23 (degree: 4)
template <> struct vnode_index<2, 23, 0> { static const int value = 0; };
template <> struct vnode_index<2, 23, 1> { static const int value = 3; };
template <> struct vnode_index<2, 23, 2> { static const int value = 5; };
template <> struct vnode_index<2, 23, 3> { static const int value = 33; };
// BG2 CHECK NODE 24 (degree: 4)
template <> struct vnode_index<2, 24, 0> { static const int value = 1; };
template <> struct vnode_index<2, 24, 1> { static const int value = 2; };
template <> struct vnode_index<2, 24, 2> { static const int value = 9; };
template <> struct vnode_index<2, 24, 3> { static const int value = 34; };
// BG2 CHECK NODE 25 (degree: 3)
template <> struct vnode_index<2, 25, 0> { static const int value = 0; };
template <> struct vnode_index<2, 25, 1> { static const int value = 5; };
template <> struct vnode_index<2, 25, 2> { static const int value = 35; };
// BG2 CHECK NODE 26 (degree: 5)
template <> struct vnode_index<2, 26, 0> { static const int value = 2; };
template <> struct vnode_index<2, 26, 1> { static const int value = 7; };
template <> struct vnode_index<2, 26, 2> { static const int value = 12; };
template <> struct vnode_index<2, 26, 3> { static const int value = 13; };
template <> struct vnode_index<2, 26, 4> { static const int value = 36; };
// BG2 CHECK NODE 27 (degree: 3)
template <> struct vnode_index<2, 27, 0> { static const int value = 0; };
template <> struct vnode_index<2, 27, 1> { static const int value = 6; };
template <> struct vnode_index<2, 27, 2> { static const int value = 37; };
// BG2 CHECK NODE 28 (degree: 4)
template <> struct vnode_index<2, 28, 0> { static const int value = 1; };
template <> struct vnode_index<2, 28, 1> { static const int value = 2; };
template <> struct vnode_index<2, 28, 2> { static const int value = 5; };
template <> struct vnode_index<2, 28, 3> { static const int value = 38; };
// BG2 CHECK NODE 29 (degree: 3)
template <> struct vnode_index<2, 29, 0> { static const int value = 0; };
template <> struct vnode_index<2, 29, 1> { static const int value = 4; };
template <> struct vnode_index<2, 29, 2> { static const int value = 39; };
// BG2 CHECK NODE 30 (degree: 5)
template <> struct vnode_index<2, 30, 0> { static const int value = 2; };
template <> struct vnode_index<2, 30, 1> { static const int value = 5; };
template <> struct vnode_index<2, 30, 2> { static const int value = 7; };
template <> struct vnode_index<2, 30, 3> { static const int value = 9; };
template <> struct vnode_index<2, 30, 4> { static const int value = 40; };
// BG2 CHECK NODE 31 (degree: 3)
template <> struct vnode_index<2, 31, 0> { static const int value = 1; };
template <> struct vnode_index<2, 31, 1> { static const int value = 13; };
template <> struct vnode_index<2, 31, 2> { static const int value = 41; };
// BG2 CHECK NODE 32 (degree: 4)
template <> struct vnode_index<2, 32, 0> { static const int value = 0; };
template <> struct vnode_index<2, 32, 1> { static const int value = 5; };
template <> struct vnode_index<2, 32, 2> { static const int value = 12; };
template <> struct vnode_index<2, 32, 3> { static const int value = 42; };
// BG2 CHECK NODE 33 (degree: 4)
template <> struct vnode_index<2, 33, 0> { static const int value = 2; };
template <> struct vnode_index<2, 33, 1> { static const int value = 7; };
template <> struct vnode_index<2, 33, 2> { static const int value = 10; };
template <> struct vnode_index<2, 33, 3> { static const int value = 43; };
// BG2 CHECK NODE 34 (degree: 4)
template <> struct vnode_index<2, 34, 0> { static const int value = 0; };
template <> struct vnode_index<2, 34, 1> { static const int value = 12; };
template <> struct vnode_index<2, 34, 2> { static const int value = 13; };
template <> struct vnode_index<2, 34, 3> { static const int value = 44; };
// BG2 CHECK NODE 35 (degree: 4)
template <> struct vnode_index<2, 35, 0> { static const int value = 1; };
template <> struct vnode_index<2, 35, 1> { static const int value = 5; };
template <> struct vnode_index<2, 35, 2> { static const int value = 11; };
template <> struct vnode_index<2, 35, 3> { static const int value = 45; };
// BG2 CHECK NODE 36 (degree: 4)
template <> struct vnode_index<2, 36, 0> { static const int value = 0; };
template <> struct vnode_index<2, 36, 1> { static const int value = 2; };
template <> struct vnode_index<2, 36, 2> { static const int value = 7; };
template <> struct vnode_index<2, 36, 3> { static const int value = 46; };
// BG2 CHECK NODE 37 (degree: 3)
template <> struct vnode_index<2, 37, 0> { static const int value = 10; };
template <> struct vnode_index<2, 37, 1> { static const int value = 13; };
template <> struct vnode_index<2, 37, 2> { static const int value = 47; };
// BG2 CHECK NODE 38 (degree: 4)
template <> struct vnode_index<2, 38, 0> { static const int value = 1; };
template <> struct vnode_index<2, 38, 1> { static const int value = 5; };
template <> struct vnode_index<2, 38, 2> { static const int value = 11; };
template <> struct vnode_index<2, 38, 3> { static const int value = 48; };
// BG2 CHECK NODE 39 (degree: 4)
template <> struct vnode_index<2, 39, 0> { static const int value = 0; };
template <> struct vnode_index<2, 39, 1> { static const int value = 7; };
template <> struct vnode_index<2, 39, 2> { static const int value = 12; };
template <> struct vnode_index<2, 39, 3> { static const int value = 49; };
// BG2 CHECK NODE 40 (degree: 4)
template <> struct vnode_index<2, 40, 0> { static const int value = 2; };
template <> struct vnode_index<2, 40, 1> { static const int value = 10; };
template <> struct vnode_index<2, 40, 2> { static const int value = 13; };
template <> struct vnode_index<2, 40, 3> { static const int value = 50; };
// BG2 CHECK NODE 41 (degree: 4)
template <> struct vnode_index<2, 41, 0> { static const int value = 1; };
template <> struct vnode_index<2, 41, 1> { static const int value = 5; };
template <> struct vnode_index<2, 41, 2> { static const int value = 11; };
template <> struct vnode_index<2, 41, 3> { static const int value = 51; };


////////////////////////////////////////////////////////////////////////
// vnode_shift
// Variable node shift values as a function of base graph,
// index set, check node index, and index within the row.
// See Tables 5.3.2-2 and 5.3.2-3, 3GPP 38.212
template <int BG, int ILS, int CHECK_NODE, int INDEX> struct vnode_shift;
// ILS = 0, CHECK_NODE = 0
template <> struct vnode_shift<1, 0, 0,  0> { static const int value =  250; };
template <> struct vnode_shift<1, 0, 0,  1> { static const int value =  69; };
template <> struct vnode_shift<1, 0, 0,  2> { static const int value =  226; };
template <> struct vnode_shift<1, 0, 0,  3> { static const int value =  159; };
template <> struct vnode_shift<1, 0, 0,  4> { static const int value =  100; };
template <> struct vnode_shift<1, 0, 0,  5> { static const int value =  10; };
template <> struct vnode_shift<1, 0, 0,  6> { static const int value =  59; };
template <> struct vnode_shift<1, 0, 0,  7> { static const int value =  229; };
template <> struct vnode_shift<1, 0, 0,  8> { static const int value =  110; };
template <> struct vnode_shift<1, 0, 0,  9> { static const int value =  191; };
template <> struct vnode_shift<1, 0, 0,  10> { static const int value =  9; };
template <> struct vnode_shift<1, 0, 0,  11> { static const int value =  195; };
template <> struct vnode_shift<1, 0, 0,  12> { static const int value =  23; };
template <> struct vnode_shift<1, 0, 0,  13> { static const int value =  190; };
template <> struct vnode_shift<1, 0, 0,  14> { static const int value =  35; };
template <> struct vnode_shift<1, 0, 0,  15> { static const int value =  239; };
template <> struct vnode_shift<1, 0, 0,  16> { static const int value =  31; };
template <> struct vnode_shift<1, 0, 0,  17> { static const int value =  1; };
template <> struct vnode_shift<1, 0, 0,  18> { static const int value =  0; };
// ILS = 0, CHECK_NODE = 1
template <> struct vnode_shift<1, 0, 1,  0> { static const int value =  2; };
template <> struct vnode_shift<1, 0, 1,  1> { static const int value =  239; };
template <> struct vnode_shift<1, 0, 1,  2> { static const int value =  117; };
template <> struct vnode_shift<1, 0, 1,  3> { static const int value =  124; };
template <> struct vnode_shift<1, 0, 1,  4> { static const int value =  71; };
template <> struct vnode_shift<1, 0, 1,  5> { static const int value =  222; };
template <> struct vnode_shift<1, 0, 1,  6> { static const int value =  104; };
template <> struct vnode_shift<1, 0, 1,  7> { static const int value =  173; };
template <> struct vnode_shift<1, 0, 1,  8> { static const int value =  220; };
template <> struct vnode_shift<1, 0, 1,  9> { static const int value =  102; };
template <> struct vnode_shift<1, 0, 1,  10> { static const int value =  109; };
template <> struct vnode_shift<1, 0, 1,  11> { static const int value =  132; };
template <> struct vnode_shift<1, 0, 1,  12> { static const int value =  142; };
template <> struct vnode_shift<1, 0, 1,  13> { static const int value =  155; };
template <> struct vnode_shift<1, 0, 1,  14> { static const int value =  255; };
template <> struct vnode_shift<1, 0, 1,  15> { static const int value =  28; };
template <> struct vnode_shift<1, 0, 1,  16> { static const int value =  0; };
template <> struct vnode_shift<1, 0, 1,  17> { static const int value =  0; };
template <> struct vnode_shift<1, 0, 1,  18> { static const int value =  0; };
// ILS = 0, CHECK_NODE = 2
template <> struct vnode_shift<1, 0, 2,  0> { static const int value =  106; };
template <> struct vnode_shift<1, 0, 2,  1> { static const int value =  111; };
template <> struct vnode_shift<1, 0, 2,  2> { static const int value =  185; };
template <> struct vnode_shift<1, 0, 2,  3> { static const int value =  63; };
template <> struct vnode_shift<1, 0, 2,  4> { static const int value =  117; };
template <> struct vnode_shift<1, 0, 2,  5> { static const int value =  93; };
template <> struct vnode_shift<1, 0, 2,  6> { static const int value =  229; };
template <> struct vnode_shift<1, 0, 2,  7> { static const int value =  177; };
template <> struct vnode_shift<1, 0, 2,  8> { static const int value =  95; };
template <> struct vnode_shift<1, 0, 2,  9> { static const int value =  39; };
template <> struct vnode_shift<1, 0, 2,  10> { static const int value =  142; };
template <> struct vnode_shift<1, 0, 2,  11> { static const int value =  225; };
template <> struct vnode_shift<1, 0, 2,  12> { static const int value =  225; };
template <> struct vnode_shift<1, 0, 2,  13> { static const int value =  245; };
template <> struct vnode_shift<1, 0, 2,  14> { static const int value =  205; };
template <> struct vnode_shift<1, 0, 2,  15> { static const int value =  251; };
template <> struct vnode_shift<1, 0, 2,  16> { static const int value =  117; };
template <> struct vnode_shift<1, 0, 2,  17> { static const int value =  0; };
template <> struct vnode_shift<1, 0, 2,  18> { static const int value =  0; };
// ILS = 0, CHECK_NODE = 3
template <> struct vnode_shift<1, 0, 3,  0> { static const int value =  121; };
template <> struct vnode_shift<1, 0, 3,  1> { static const int value =  89; };
template <> struct vnode_shift<1, 0, 3,  2> { static const int value =  84; };
template <> struct vnode_shift<1, 0, 3,  3> { static const int value =  20; };
template <> struct vnode_shift<1, 0, 3,  4> { static const int value =  150; };
template <> struct vnode_shift<1, 0, 3,  5> { static const int value =  131; };
template <> struct vnode_shift<1, 0, 3,  6> { static const int value =  243; };
template <> struct vnode_shift<1, 0, 3,  7> { static const int value =  136; };
template <> struct vnode_shift<1, 0, 3,  8> { static const int value =  86; };
template <> struct vnode_shift<1, 0, 3,  9> { static const int value =  246; };
template <> struct vnode_shift<1, 0, 3,  10> { static const int value =  219; };
template <> struct vnode_shift<1, 0, 3,  11> { static const int value =  211; };
template <> struct vnode_shift<1, 0, 3,  12> { static const int value =  240; };
template <> struct vnode_shift<1, 0, 3,  13> { static const int value =  76; };
template <> struct vnode_shift<1, 0, 3,  14> { static const int value =  244; };
template <> struct vnode_shift<1, 0, 3,  15> { static const int value =  144; };
template <> struct vnode_shift<1, 0, 3,  16> { static const int value =  12; };
template <> struct vnode_shift<1, 0, 3,  17> { static const int value =  1; };
template <> struct vnode_shift<1, 0, 3,  18> { static const int value =  0; };
// ILS = 0, CHECK_NODE = 4
template <> struct vnode_shift<1, 0, 4,  0> { static const int value =  157; };
template <> struct vnode_shift<1, 0, 4,  1> { static const int value =  102; };
template <> struct vnode_shift<1, 0, 4,  2> { static const int value =  0; };
// ILS = 0, CHECK_NODE = 5
template <> struct vnode_shift<1, 0, 5,  0> { static const int value =  205; };
template <> struct vnode_shift<1, 0, 5,  1> { static const int value =  236; };
template <> struct vnode_shift<1, 0, 5,  2> { static const int value =  194; };
template <> struct vnode_shift<1, 0, 5,  3> { static const int value =  231; };
template <> struct vnode_shift<1, 0, 5,  4> { static const int value =  28; };
template <> struct vnode_shift<1, 0, 5,  5> { static const int value =  123; };
template <> struct vnode_shift<1, 0, 5,  6> { static const int value =  115; };
template <> struct vnode_shift<1, 0, 5,  7> { static const int value =  0; };
// ILS = 0, CHECK_NODE = 6
template <> struct vnode_shift<1, 0, 6,  0> { static const int value =  183; };
template <> struct vnode_shift<1, 0, 6,  1> { static const int value =  22; };
template <> struct vnode_shift<1, 0, 6,  2> { static const int value =  28; };
template <> struct vnode_shift<1, 0, 6,  3> { static const int value =  67; };
template <> struct vnode_shift<1, 0, 6,  4> { static const int value =  244; };
template <> struct vnode_shift<1, 0, 6,  5> { static const int value =  11; };
template <> struct vnode_shift<1, 0, 6,  6> { static const int value =  157; };
template <> struct vnode_shift<1, 0, 6,  7> { static const int value =  211; };
template <> struct vnode_shift<1, 0, 6,  8> { static const int value =  0; };
// ILS = 0, CHECK_NODE = 7
template <> struct vnode_shift<1, 0, 7,  0> { static const int value =  220; };
template <> struct vnode_shift<1, 0, 7,  1> { static const int value =  44; };
template <> struct vnode_shift<1, 0, 7,  2> { static const int value =  159; };
template <> struct vnode_shift<1, 0, 7,  3> { static const int value =  31; };
template <> struct vnode_shift<1, 0, 7,  4> { static const int value =  167; };
template <> struct vnode_shift<1, 0, 7,  5> { static const int value =  104; };
template <> struct vnode_shift<1, 0, 7,  6> { static const int value =  0; };
// ILS = 0, CHECK_NODE = 8
template <> struct vnode_shift<1, 0, 8,  0> { static const int value =  112; };
template <> struct vnode_shift<1, 0, 8,  1> { static const int value =  4; };
template <> struct vnode_shift<1, 0, 8,  2> { static const int value =  7; };
template <> struct vnode_shift<1, 0, 8,  3> { static const int value =  211; };
template <> struct vnode_shift<1, 0, 8,  4> { static const int value =  102; };
template <> struct vnode_shift<1, 0, 8,  5> { static const int value =  164; };
template <> struct vnode_shift<1, 0, 8,  6> { static const int value =  109; };
template <> struct vnode_shift<1, 0, 8,  7> { static const int value =  241; };
template <> struct vnode_shift<1, 0, 8,  8> { static const int value =  90; };
template <> struct vnode_shift<1, 0, 8,  9> { static const int value =  0; };
// ILS = 0, CHECK_NODE = 9
template <> struct vnode_shift<1, 0, 9,  0> { static const int value =  103; };
template <> struct vnode_shift<1, 0, 9,  1> { static const int value =  182; };
template <> struct vnode_shift<1, 0, 9,  2> { static const int value =  109; };
template <> struct vnode_shift<1, 0, 9,  3> { static const int value =  21; };
template <> struct vnode_shift<1, 0, 9,  4> { static const int value =  142; };
template <> struct vnode_shift<1, 0, 9,  5> { static const int value =  14; };
template <> struct vnode_shift<1, 0, 9,  6> { static const int value =  61; };
template <> struct vnode_shift<1, 0, 9,  7> { static const int value =  216; };
template <> struct vnode_shift<1, 0, 9,  8> { static const int value =  0; };
// ILS = 0, CHECK_NODE = 10
template <> struct vnode_shift<1, 0, 10,  0> { static const int value =  98; };
template <> struct vnode_shift<1, 0, 10,  1> { static const int value =  149; };
template <> struct vnode_shift<1, 0, 10,  2> { static const int value =  167; };
template <> struct vnode_shift<1, 0, 10,  3> { static const int value =  160; };
template <> struct vnode_shift<1, 0, 10,  4> { static const int value =  49; };
template <> struct vnode_shift<1, 0, 10,  5> { static const int value =  58; };
template <> struct vnode_shift<1, 0, 10,  6> { static const int value =  0; };
// ILS = 0, CHECK_NODE = 11
template <> struct vnode_shift<1, 0, 11,  0> { static const int value =  77; };
template <> struct vnode_shift<1, 0, 11,  1> { static const int value =  41; };
template <> struct vnode_shift<1, 0, 11,  2> { static const int value =  83; };
template <> struct vnode_shift<1, 0, 11,  3> { static const int value =  182; };
template <> struct vnode_shift<1, 0, 11,  4> { static const int value =  78; };
template <> struct vnode_shift<1, 0, 11,  5> { static const int value =  252; };
template <> struct vnode_shift<1, 0, 11,  6> { static const int value =  22; };
template <> struct vnode_shift<1, 0, 11,  7> { static const int value =  0; };
// ILS = 0, CHECK_NODE = 12
template <> struct vnode_shift<1, 0, 12,  0> { static const int value =  160; };
template <> struct vnode_shift<1, 0, 12,  1> { static const int value =  42; };
template <> struct vnode_shift<1, 0, 12,  2> { static const int value =  21; };
template <> struct vnode_shift<1, 0, 12,  3> { static const int value =  32; };
template <> struct vnode_shift<1, 0, 12,  4> { static const int value =  234; };
template <> struct vnode_shift<1, 0, 12,  5> { static const int value =  7; };
template <> struct vnode_shift<1, 0, 12,  6> { static const int value =  0; };
// ILS = 0, CHECK_NODE = 13
template <> struct vnode_shift<1, 0, 13,  0> { static const int value =  177; };
template <> struct vnode_shift<1, 0, 13,  1> { static const int value =  248; };
template <> struct vnode_shift<1, 0, 13,  2> { static const int value =  151; };
template <> struct vnode_shift<1, 0, 13,  3> { static const int value =  185; };
template <> struct vnode_shift<1, 0, 13,  4> { static const int value =  62; };
template <> struct vnode_shift<1, 0, 13,  5> { static const int value =  0; };
// ILS = 0, CHECK_NODE = 14
template <> struct vnode_shift<1, 0, 14,  0> { static const int value =  206; };
template <> struct vnode_shift<1, 0, 14,  1> { static const int value =  55; };
template <> struct vnode_shift<1, 0, 14,  2> { static const int value =  206; };
template <> struct vnode_shift<1, 0, 14,  3> { static const int value =  127; };
template <> struct vnode_shift<1, 0, 14,  4> { static const int value =  16; };
template <> struct vnode_shift<1, 0, 14,  5> { static const int value =  229; };
template <> struct vnode_shift<1, 0, 14,  6> { static const int value =  0; };
// ILS = 0, CHECK_NODE = 15
template <> struct vnode_shift<1, 0, 15,  0> { static const int value =  40; };
template <> struct vnode_shift<1, 0, 15,  1> { static const int value =  96; };
template <> struct vnode_shift<1, 0, 15,  2> { static const int value =  65; };
template <> struct vnode_shift<1, 0, 15,  3> { static const int value =  63; };
template <> struct vnode_shift<1, 0, 15,  4> { static const int value =  75; };
template <> struct vnode_shift<1, 0, 15,  5> { static const int value =  179; };
template <> struct vnode_shift<1, 0, 15,  6> { static const int value =  0; };
// ILS = 0, CHECK_NODE = 16
template <> struct vnode_shift<1, 0, 16,  0> { static const int value =  64; };
template <> struct vnode_shift<1, 0, 16,  1> { static const int value =  49; };
template <> struct vnode_shift<1, 0, 16,  2> { static const int value =  49; };
template <> struct vnode_shift<1, 0, 16,  3> { static const int value =  51; };
template <> struct vnode_shift<1, 0, 16,  4> { static const int value =  154; };
template <> struct vnode_shift<1, 0, 16,  5> { static const int value =  0; };
// ILS = 0, CHECK_NODE = 17
template <> struct vnode_shift<1, 0, 17,  0> { static const int value =  7; };
template <> struct vnode_shift<1, 0, 17,  1> { static const int value =  164; };
template <> struct vnode_shift<1, 0, 17,  2> { static const int value =  59; };
template <> struct vnode_shift<1, 0, 17,  3> { static const int value =  1; };
template <> struct vnode_shift<1, 0, 17,  4> { static const int value =  144; };
template <> struct vnode_shift<1, 0, 17,  5> { static const int value =  0; };
// ILS = 0, CHECK_NODE = 18
template <> struct vnode_shift<1, 0, 18,  0> { static const int value =  42; };
template <> struct vnode_shift<1, 0, 18,  1> { static const int value =  233; };
template <> struct vnode_shift<1, 0, 18,  2> { static const int value =  8; };
template <> struct vnode_shift<1, 0, 18,  3> { static const int value =  155; };
template <> struct vnode_shift<1, 0, 18,  4> { static const int value =  147; };
template <> struct vnode_shift<1, 0, 18,  5> { static const int value =  0; };
// ILS = 0, CHECK_NODE = 19
template <> struct vnode_shift<1, 0, 19,  0> { static const int value =  60; };
template <> struct vnode_shift<1, 0, 19,  1> { static const int value =  73; };
template <> struct vnode_shift<1, 0, 19,  2> { static const int value =  72; };
template <> struct vnode_shift<1, 0, 19,  3> { static const int value =  127; };
template <> struct vnode_shift<1, 0, 19,  4> { static const int value =  224; };
template <> struct vnode_shift<1, 0, 19,  5> { static const int value =  0; };
// ILS = 0, CHECK_NODE = 20
template <> struct vnode_shift<1, 0, 20,  0> { static const int value =  151; };
template <> struct vnode_shift<1, 0, 20,  1> { static const int value =  186; };
template <> struct vnode_shift<1, 0, 20,  2> { static const int value =  217; };
template <> struct vnode_shift<1, 0, 20,  3> { static const int value =  47; };
template <> struct vnode_shift<1, 0, 20,  4> { static const int value =  160; };
template <> struct vnode_shift<1, 0, 20,  5> { static const int value =  0; };
// ILS = 0, CHECK_NODE = 21
template <> struct vnode_shift<1, 0, 21,  0> { static const int value =  249; };
template <> struct vnode_shift<1, 0, 21,  1> { static const int value =  121; };
template <> struct vnode_shift<1, 0, 21,  2> { static const int value =  109; };
template <> struct vnode_shift<1, 0, 21,  3> { static const int value =  131; };
template <> struct vnode_shift<1, 0, 21,  4> { static const int value =  171; };
template <> struct vnode_shift<1, 0, 21,  5> { static const int value =  0; };
// ILS = 0, CHECK_NODE = 22
template <> struct vnode_shift<1, 0, 22,  0> { static const int value =  64; };
template <> struct vnode_shift<1, 0, 22,  1> { static const int value =  142; };
template <> struct vnode_shift<1, 0, 22,  2> { static const int value =  188; };
template <> struct vnode_shift<1, 0, 22,  3> { static const int value =  158; };
template <> struct vnode_shift<1, 0, 22,  4> { static const int value =  0; };
// ILS = 0, CHECK_NODE = 23
template <> struct vnode_shift<1, 0, 23,  0> { static const int value =  156; };
template <> struct vnode_shift<1, 0, 23,  1> { static const int value =  147; };
template <> struct vnode_shift<1, 0, 23,  2> { static const int value =  170; };
template <> struct vnode_shift<1, 0, 23,  3> { static const int value =  152; };
template <> struct vnode_shift<1, 0, 23,  4> { static const int value =  0; };
// ILS = 0, CHECK_NODE = 24
template <> struct vnode_shift<1, 0, 24,  0> { static const int value =  112; };
template <> struct vnode_shift<1, 0, 24,  1> { static const int value =  86; };
template <> struct vnode_shift<1, 0, 24,  2> { static const int value =  236; };
template <> struct vnode_shift<1, 0, 24,  3> { static const int value =  116; };
template <> struct vnode_shift<1, 0, 24,  4> { static const int value =  222; };
template <> struct vnode_shift<1, 0, 24,  5> { static const int value =  0; };
// ILS = 0, CHECK_NODE = 25
template <> struct vnode_shift<1, 0, 25,  0> { static const int value =  23; };
template <> struct vnode_shift<1, 0, 25,  1> { static const int value =  136; };
template <> struct vnode_shift<1, 0, 25,  2> { static const int value =  116; };
template <> struct vnode_shift<1, 0, 25,  3> { static const int value =  182; };
template <> struct vnode_shift<1, 0, 25,  4> { static const int value =  0; };
// ILS = 0, CHECK_NODE = 26
template <> struct vnode_shift<1, 0, 26,  0> { static const int value =  195; };
template <> struct vnode_shift<1, 0, 26,  1> { static const int value =  243; };
template <> struct vnode_shift<1, 0, 26,  2> { static const int value =  215; };
template <> struct vnode_shift<1, 0, 26,  3> { static const int value =  61; };
template <> struct vnode_shift<1, 0, 26,  4> { static const int value =  0; };
// ILS = 0, CHECK_NODE = 27
template <> struct vnode_shift<1, 0, 27,  0> { static const int value =  25; };
template <> struct vnode_shift<1, 0, 27,  1> { static const int value =  104; };
template <> struct vnode_shift<1, 0, 27,  2> { static const int value =  194; };
template <> struct vnode_shift<1, 0, 27,  3> { static const int value =  0; };
// ILS = 0, CHECK_NODE = 28
template <> struct vnode_shift<1, 0, 28,  0> { static const int value =  128; };
template <> struct vnode_shift<1, 0, 28,  1> { static const int value =  165; };
template <> struct vnode_shift<1, 0, 28,  2> { static const int value =  181; };
template <> struct vnode_shift<1, 0, 28,  3> { static const int value =  63; };
template <> struct vnode_shift<1, 0, 28,  4> { static const int value =  0; };
// ILS = 0, CHECK_NODE = 29
template <> struct vnode_shift<1, 0, 29,  0> { static const int value =  86; };
template <> struct vnode_shift<1, 0, 29,  1> { static const int value =  236; };
template <> struct vnode_shift<1, 0, 29,  2> { static const int value =  84; };
template <> struct vnode_shift<1, 0, 29,  3> { static const int value =  6; };
template <> struct vnode_shift<1, 0, 29,  4> { static const int value =  0; };
// ILS = 0, CHECK_NODE = 30
template <> struct vnode_shift<1, 0, 30,  0> { static const int value =  216; };
template <> struct vnode_shift<1, 0, 30,  1> { static const int value =  73; };
template <> struct vnode_shift<1, 0, 30,  2> { static const int value =  120; };
template <> struct vnode_shift<1, 0, 30,  3> { static const int value =  9; };
template <> struct vnode_shift<1, 0, 30,  4> { static const int value =  0; };
// ILS = 0, CHECK_NODE = 31
template <> struct vnode_shift<1, 0, 31,  0> { static const int value =  95; };
template <> struct vnode_shift<1, 0, 31,  1> { static const int value =  177; };
template <> struct vnode_shift<1, 0, 31,  2> { static const int value =  172; };
template <> struct vnode_shift<1, 0, 31,  3> { static const int value =  61; };
template <> struct vnode_shift<1, 0, 31,  4> { static const int value =  0; };
// ILS = 0, CHECK_NODE = 32
template <> struct vnode_shift<1, 0, 32,  0> { static const int value =  221; };
template <> struct vnode_shift<1, 0, 32,  1> { static const int value =  112; };
template <> struct vnode_shift<1, 0, 32,  2> { static const int value =  199; };
template <> struct vnode_shift<1, 0, 32,  3> { static const int value =  121; };
template <> struct vnode_shift<1, 0, 32,  4> { static const int value =  0; };
// ILS = 0, CHECK_NODE = 33
template <> struct vnode_shift<1, 0, 33,  0> { static const int value =  2; };
template <> struct vnode_shift<1, 0, 33,  1> { static const int value =  187; };
template <> struct vnode_shift<1, 0, 33,  2> { static const int value =  41; };
template <> struct vnode_shift<1, 0, 33,  3> { static const int value =  211; };
template <> struct vnode_shift<1, 0, 33,  4> { static const int value =  0; };
// ILS = 0, CHECK_NODE = 34
template <> struct vnode_shift<1, 0, 34,  0> { static const int value =  127; };
template <> struct vnode_shift<1, 0, 34,  1> { static const int value =  167; };
template <> struct vnode_shift<1, 0, 34,  2> { static const int value =  164; };
template <> struct vnode_shift<1, 0, 34,  3> { static const int value =  159; };
template <> struct vnode_shift<1, 0, 34,  4> { static const int value =  0; };
// ILS = 0, CHECK_NODE = 35
template <> struct vnode_shift<1, 0, 35,  0> { static const int value =  161; };
template <> struct vnode_shift<1, 0, 35,  1> { static const int value =  197; };
template <> struct vnode_shift<1, 0, 35,  2> { static const int value =  207; };
template <> struct vnode_shift<1, 0, 35,  3> { static const int value =  103; };
template <> struct vnode_shift<1, 0, 35,  4> { static const int value =  0; };
// ILS = 0, CHECK_NODE = 36
template <> struct vnode_shift<1, 0, 36,  0> { static const int value =  37; };
template <> struct vnode_shift<1, 0, 36,  1> { static const int value =  105; };
template <> struct vnode_shift<1, 0, 36,  2> { static const int value =  51; };
template <> struct vnode_shift<1, 0, 36,  3> { static const int value =  120; };
template <> struct vnode_shift<1, 0, 36,  4> { static const int value =  0; };
// ILS = 0, CHECK_NODE = 37
template <> struct vnode_shift<1, 0, 37,  0> { static const int value =  198; };
template <> struct vnode_shift<1, 0, 37,  1> { static const int value =  220; };
template <> struct vnode_shift<1, 0, 37,  2> { static const int value =  122; };
template <> struct vnode_shift<1, 0, 37,  3> { static const int value =  0; };
// ILS = 0, CHECK_NODE = 38
template <> struct vnode_shift<1, 0, 38,  0> { static const int value =  167; };
template <> struct vnode_shift<1, 0, 38,  1> { static const int value =  151; };
template <> struct vnode_shift<1, 0, 38,  2> { static const int value =  157; };
template <> struct vnode_shift<1, 0, 38,  3> { static const int value =  163; };
template <> struct vnode_shift<1, 0, 38,  4> { static const int value =  0; };
// ILS = 0, CHECK_NODE = 39
template <> struct vnode_shift<1, 0, 39,  0> { static const int value =  173; };
template <> struct vnode_shift<1, 0, 39,  1> { static const int value =  139; };
template <> struct vnode_shift<1, 0, 39,  2> { static const int value =  149; };
template <> struct vnode_shift<1, 0, 39,  3> { static const int value =  0; };
template <> struct vnode_shift<1, 0, 39,  4> { static const int value =  0; };
// ILS = 0, CHECK_NODE = 40
template <> struct vnode_shift<1, 0, 40,  0> { static const int value =  157; };
template <> struct vnode_shift<1, 0, 40,  1> { static const int value =  137; };
template <> struct vnode_shift<1, 0, 40,  2> { static const int value =  149; };
template <> struct vnode_shift<1, 0, 40,  3> { static const int value =  0; };
// ILS = 0, CHECK_NODE = 41
template <> struct vnode_shift<1, 0, 41,  0> { static const int value =  167; };
template <> struct vnode_shift<1, 0, 41,  1> { static const int value =  173; };
template <> struct vnode_shift<1, 0, 41,  2> { static const int value =  139; };
template <> struct vnode_shift<1, 0, 41,  3> { static const int value =  151; };
template <> struct vnode_shift<1, 0, 41,  4> { static const int value =  0; };
// ILS = 0, CHECK_NODE = 42
template <> struct vnode_shift<1, 0, 42,  0> { static const int value =  149; };
template <> struct vnode_shift<1, 0, 42,  1> { static const int value =  157; };
template <> struct vnode_shift<1, 0, 42,  2> { static const int value =  137; };
template <> struct vnode_shift<1, 0, 42,  3> { static const int value =  0; };
// ILS = 0, CHECK_NODE = 43
template <> struct vnode_shift<1, 0, 43,  0> { static const int value =  151; };
template <> struct vnode_shift<1, 0, 43,  1> { static const int value =  163; };
template <> struct vnode_shift<1, 0, 43,  2> { static const int value =  173; };
template <> struct vnode_shift<1, 0, 43,  3> { static const int value =  139; };
template <> struct vnode_shift<1, 0, 43,  4> { static const int value =  0; };
// ILS = 0, CHECK_NODE = 44
template <> struct vnode_shift<1, 0, 44,  0> { static const int value =  139; };
template <> struct vnode_shift<1, 0, 44,  1> { static const int value =  157; };
template <> struct vnode_shift<1, 0, 44,  2> { static const int value =  163; };
template <> struct vnode_shift<1, 0, 44,  3> { static const int value =  173; };
template <> struct vnode_shift<1, 0, 44,  4> { static const int value =  0; };
// ILS = 0, CHECK_NODE = 45
template <> struct vnode_shift<1, 0, 45,  0> { static const int value =  149; };
template <> struct vnode_shift<1, 0, 45,  1> { static const int value =  151; };
template <> struct vnode_shift<1, 0, 45,  2> { static const int value =  167; };
template <> struct vnode_shift<1, 0, 45,  3> { static const int value =  0; };
// ILS = 1, CHECK_NODE = 0
template <> struct vnode_shift<1, 1, 0,  0> { static const int value =  307; };
template <> struct vnode_shift<1, 1, 0,  1> { static const int value =  19; };
template <> struct vnode_shift<1, 1, 0,  2> { static const int value =  50; };
template <> struct vnode_shift<1, 1, 0,  3> { static const int value =  369; };
template <> struct vnode_shift<1, 1, 0,  4> { static const int value =  181; };
template <> struct vnode_shift<1, 1, 0,  5> { static const int value =  216; };
template <> struct vnode_shift<1, 1, 0,  6> { static const int value =  317; };
template <> struct vnode_shift<1, 1, 0,  7> { static const int value =  288; };
template <> struct vnode_shift<1, 1, 0,  8> { static const int value =  109; };
template <> struct vnode_shift<1, 1, 0,  9> { static const int value =  17; };
template <> struct vnode_shift<1, 1, 0,  10> { static const int value =  357; };
template <> struct vnode_shift<1, 1, 0,  11> { static const int value =  215; };
template <> struct vnode_shift<1, 1, 0,  12> { static const int value =  106; };
template <> struct vnode_shift<1, 1, 0,  13> { static const int value =  242; };
template <> struct vnode_shift<1, 1, 0,  14> { static const int value =  180; };
template <> struct vnode_shift<1, 1, 0,  15> { static const int value =  330; };
template <> struct vnode_shift<1, 1, 0,  16> { static const int value =  346; };
template <> struct vnode_shift<1, 1, 0,  17> { static const int value =  1; };
template <> struct vnode_shift<1, 1, 0,  18> { static const int value =  0; };
// ILS = 1, CHECK_NODE = 1
template <> struct vnode_shift<1, 1, 1,  0> { static const int value =  76; };
template <> struct vnode_shift<1, 1, 1,  1> { static const int value =  76; };
template <> struct vnode_shift<1, 1, 1,  2> { static const int value =  73; };
template <> struct vnode_shift<1, 1, 1,  3> { static const int value =  288; };
template <> struct vnode_shift<1, 1, 1,  4> { static const int value =  144; };
template <> struct vnode_shift<1, 1, 1,  5> { static const int value =  331; };
template <> struct vnode_shift<1, 1, 1,  6> { static const int value =  331; };
template <> struct vnode_shift<1, 1, 1,  7> { static const int value =  178; };
template <> struct vnode_shift<1, 1, 1,  8> { static const int value =  295; };
template <> struct vnode_shift<1, 1, 1,  9> { static const int value =  342; };
template <> struct vnode_shift<1, 1, 1,  10> { static const int value =  217; };
template <> struct vnode_shift<1, 1, 1,  11> { static const int value =  99; };
template <> struct vnode_shift<1, 1, 1,  12> { static const int value =  354; };
template <> struct vnode_shift<1, 1, 1,  13> { static const int value =  114; };
template <> struct vnode_shift<1, 1, 1,  14> { static const int value =  331; };
template <> struct vnode_shift<1, 1, 1,  15> { static const int value =  112; };
template <> struct vnode_shift<1, 1, 1,  16> { static const int value =  0; };
template <> struct vnode_shift<1, 1, 1,  17> { static const int value =  0; };
template <> struct vnode_shift<1, 1, 1,  18> { static const int value =  0; };
// ILS = 1, CHECK_NODE = 2
template <> struct vnode_shift<1, 1, 2,  0> { static const int value =  205; };
template <> struct vnode_shift<1, 1, 2,  1> { static const int value =  250; };
template <> struct vnode_shift<1, 1, 2,  2> { static const int value =  328; };
template <> struct vnode_shift<1, 1, 2,  3> { static const int value =  332; };
template <> struct vnode_shift<1, 1, 2,  4> { static const int value =  256; };
template <> struct vnode_shift<1, 1, 2,  5> { static const int value =  161; };
template <> struct vnode_shift<1, 1, 2,  6> { static const int value =  267; };
template <> struct vnode_shift<1, 1, 2,  7> { static const int value =  160; };
template <> struct vnode_shift<1, 1, 2,  8> { static const int value =  63; };
template <> struct vnode_shift<1, 1, 2,  9> { static const int value =  129; };
template <> struct vnode_shift<1, 1, 2,  10> { static const int value =  200; };
template <> struct vnode_shift<1, 1, 2,  11> { static const int value =  88; };
template <> struct vnode_shift<1, 1, 2,  12> { static const int value =  53; };
template <> struct vnode_shift<1, 1, 2,  13> { static const int value =  131; };
template <> struct vnode_shift<1, 1, 2,  14> { static const int value =  240; };
template <> struct vnode_shift<1, 1, 2,  15> { static const int value =  205; };
template <> struct vnode_shift<1, 1, 2,  16> { static const int value =  13; };
template <> struct vnode_shift<1, 1, 2,  17> { static const int value =  0; };
template <> struct vnode_shift<1, 1, 2,  18> { static const int value =  0; };
// ILS = 1, CHECK_NODE = 3
template <> struct vnode_shift<1, 1, 3,  0> { static const int value =  276; };
template <> struct vnode_shift<1, 1, 3,  1> { static const int value =  87; };
template <> struct vnode_shift<1, 1, 3,  2> { static const int value =  0; };
template <> struct vnode_shift<1, 1, 3,  3> { static const int value =  275; };
template <> struct vnode_shift<1, 1, 3,  4> { static const int value =  199; };
template <> struct vnode_shift<1, 1, 3,  5> { static const int value =  153; };
template <> struct vnode_shift<1, 1, 3,  6> { static const int value =  56; };
template <> struct vnode_shift<1, 1, 3,  7> { static const int value =  132; };
template <> struct vnode_shift<1, 1, 3,  8> { static const int value =  305; };
template <> struct vnode_shift<1, 1, 3,  9> { static const int value =  231; };
template <> struct vnode_shift<1, 1, 3,  10> { static const int value =  341; };
template <> struct vnode_shift<1, 1, 3,  11> { static const int value =  212; };
template <> struct vnode_shift<1, 1, 3,  12> { static const int value =  304; };
template <> struct vnode_shift<1, 1, 3,  13> { static const int value =  300; };
template <> struct vnode_shift<1, 1, 3,  14> { static const int value =  271; };
template <> struct vnode_shift<1, 1, 3,  15> { static const int value =  39; };
template <> struct vnode_shift<1, 1, 3,  16> { static const int value =  357; };
template <> struct vnode_shift<1, 1, 3,  17> { static const int value =  1; };
template <> struct vnode_shift<1, 1, 3,  18> { static const int value =  0; };
// ILS = 1, CHECK_NODE = 4
template <> struct vnode_shift<1, 1, 4,  0> { static const int value =  332; };
template <> struct vnode_shift<1, 1, 4,  1> { static const int value =  181; };
template <> struct vnode_shift<1, 1, 4,  2> { static const int value =  0; };
// ILS = 1, CHECK_NODE = 5
template <> struct vnode_shift<1, 1, 5,  0> { static const int value =  195; };
template <> struct vnode_shift<1, 1, 5,  1> { static const int value =  14; };
template <> struct vnode_shift<1, 1, 5,  2> { static const int value =  115; };
template <> struct vnode_shift<1, 1, 5,  3> { static const int value =  166; };
template <> struct vnode_shift<1, 1, 5,  4> { static const int value =  241; };
template <> struct vnode_shift<1, 1, 5,  5> { static const int value =  51; };
template <> struct vnode_shift<1, 1, 5,  6> { static const int value =  157; };
template <> struct vnode_shift<1, 1, 5,  7> { static const int value =  0; };
// ILS = 1, CHECK_NODE = 6
template <> struct vnode_shift<1, 1, 6,  0> { static const int value =  278; };
template <> struct vnode_shift<1, 1, 6,  1> { static const int value =  257; };
template <> struct vnode_shift<1, 1, 6,  2> { static const int value =  1; };
template <> struct vnode_shift<1, 1, 6,  3> { static const int value =  351; };
template <> struct vnode_shift<1, 1, 6,  4> { static const int value =  92; };
template <> struct vnode_shift<1, 1, 6,  5> { static const int value =  253; };
template <> struct vnode_shift<1, 1, 6,  6> { static const int value =  18; };
template <> struct vnode_shift<1, 1, 6,  7> { static const int value =  225; };
template <> struct vnode_shift<1, 1, 6,  8> { static const int value =  0; };
// ILS = 1, CHECK_NODE = 7
template <> struct vnode_shift<1, 1, 7,  0> { static const int value =  9; };
template <> struct vnode_shift<1, 1, 7,  1> { static const int value =  62; };
template <> struct vnode_shift<1, 1, 7,  2> { static const int value =  316; };
template <> struct vnode_shift<1, 1, 7,  3> { static const int value =  333; };
template <> struct vnode_shift<1, 1, 7,  4> { static const int value =  290; };
template <> struct vnode_shift<1, 1, 7,  5> { static const int value =  114; };
template <> struct vnode_shift<1, 1, 7,  6> { static const int value =  0; };
// ILS = 1, CHECK_NODE = 8
template <> struct vnode_shift<1, 1, 8,  0> { static const int value =  307; };
template <> struct vnode_shift<1, 1, 8,  1> { static const int value =  179; };
template <> struct vnode_shift<1, 1, 8,  2> { static const int value =  165; };
template <> struct vnode_shift<1, 1, 8,  3> { static const int value =  18; };
template <> struct vnode_shift<1, 1, 8,  4> { static const int value =  39; };
template <> struct vnode_shift<1, 1, 8,  5> { static const int value =  224; };
template <> struct vnode_shift<1, 1, 8,  6> { static const int value =  368; };
template <> struct vnode_shift<1, 1, 8,  7> { static const int value =  67; };
template <> struct vnode_shift<1, 1, 8,  8> { static const int value =  170; };
template <> struct vnode_shift<1, 1, 8,  9> { static const int value =  0; };
// ILS = 1, CHECK_NODE = 9
template <> struct vnode_shift<1, 1, 9,  0> { static const int value =  366; };
template <> struct vnode_shift<1, 1, 9,  1> { static const int value =  232; };
template <> struct vnode_shift<1, 1, 9,  2> { static const int value =  321; };
template <> struct vnode_shift<1, 1, 9,  3> { static const int value =  133; };
template <> struct vnode_shift<1, 1, 9,  4> { static const int value =  57; };
template <> struct vnode_shift<1, 1, 9,  5> { static const int value =  303; };
template <> struct vnode_shift<1, 1, 9,  6> { static const int value =  63; };
template <> struct vnode_shift<1, 1, 9,  7> { static const int value =  82; };
template <> struct vnode_shift<1, 1, 9,  8> { static const int value =  0; };
// ILS = 1, CHECK_NODE = 10
template <> struct vnode_shift<1, 1, 10,  0> { static const int value =  101; };
template <> struct vnode_shift<1, 1, 10,  1> { static const int value =  339; };
template <> struct vnode_shift<1, 1, 10,  2> { static const int value =  274; };
template <> struct vnode_shift<1, 1, 10,  3> { static const int value =  111; };
template <> struct vnode_shift<1, 1, 10,  4> { static const int value =  383; };
template <> struct vnode_shift<1, 1, 10,  5> { static const int value =  354; };
template <> struct vnode_shift<1, 1, 10,  6> { static const int value =  0; };
// ILS = 1, CHECK_NODE = 11
template <> struct vnode_shift<1, 1, 11,  0> { static const int value =  48; };
template <> struct vnode_shift<1, 1, 11,  1> { static const int value =  102; };
template <> struct vnode_shift<1, 1, 11,  2> { static const int value =  8; };
template <> struct vnode_shift<1, 1, 11,  3> { static const int value =  47; };
template <> struct vnode_shift<1, 1, 11,  4> { static const int value =  188; };
template <> struct vnode_shift<1, 1, 11,  5> { static const int value =  334; };
template <> struct vnode_shift<1, 1, 11,  6> { static const int value =  115; };
template <> struct vnode_shift<1, 1, 11,  7> { static const int value =  0; };
// ILS = 1, CHECK_NODE = 12
template <> struct vnode_shift<1, 1, 12,  0> { static const int value =  77; };
template <> struct vnode_shift<1, 1, 12,  1> { static const int value =  186; };
template <> struct vnode_shift<1, 1, 12,  2> { static const int value =  174; };
template <> struct vnode_shift<1, 1, 12,  3> { static const int value =  232; };
template <> struct vnode_shift<1, 1, 12,  4> { static const int value =  50; };
template <> struct vnode_shift<1, 1, 12,  5> { static const int value =  74; };
template <> struct vnode_shift<1, 1, 12,  6> { static const int value =  0; };
// ILS = 1, CHECK_NODE = 13
template <> struct vnode_shift<1, 1, 13,  0> { static const int value =  313; };
template <> struct vnode_shift<1, 1, 13,  1> { static const int value =  177; };
template <> struct vnode_shift<1, 1, 13,  2> { static const int value =  266; };
template <> struct vnode_shift<1, 1, 13,  3> { static const int value =  115; };
template <> struct vnode_shift<1, 1, 13,  4> { static const int value =  370; };
template <> struct vnode_shift<1, 1, 13,  5> { static const int value =  0; };
// ILS = 1, CHECK_NODE = 14
template <> struct vnode_shift<1, 1, 14,  0> { static const int value =  142; };
template <> struct vnode_shift<1, 1, 14,  1> { static const int value =  248; };
template <> struct vnode_shift<1, 1, 14,  2> { static const int value =  137; };
template <> struct vnode_shift<1, 1, 14,  3> { static const int value =  89; };
template <> struct vnode_shift<1, 1, 14,  4> { static const int value =  347; };
template <> struct vnode_shift<1, 1, 14,  5> { static const int value =  12; };
template <> struct vnode_shift<1, 1, 14,  6> { static const int value =  0; };
// ILS = 1, CHECK_NODE = 15
template <> struct vnode_shift<1, 1, 15,  0> { static const int value =  241; };
template <> struct vnode_shift<1, 1, 15,  1> { static const int value =  2; };
template <> struct vnode_shift<1, 1, 15,  2> { static const int value =  210; };
template <> struct vnode_shift<1, 1, 15,  3> { static const int value =  318; };
template <> struct vnode_shift<1, 1, 15,  4> { static const int value =  55; };
template <> struct vnode_shift<1, 1, 15,  5> { static const int value =  269; };
template <> struct vnode_shift<1, 1, 15,  6> { static const int value =  0; };
// ILS = 1, CHECK_NODE = 16
template <> struct vnode_shift<1, 1, 16,  0> { static const int value =  13; };
template <> struct vnode_shift<1, 1, 16,  1> { static const int value =  338; };
template <> struct vnode_shift<1, 1, 16,  2> { static const int value =  57; };
template <> struct vnode_shift<1, 1, 16,  3> { static const int value =  289; };
template <> struct vnode_shift<1, 1, 16,  4> { static const int value =  57; };
template <> struct vnode_shift<1, 1, 16,  5> { static const int value =  0; };
// ILS = 1, CHECK_NODE = 17
template <> struct vnode_shift<1, 1, 17,  0> { static const int value =  260; };
template <> struct vnode_shift<1, 1, 17,  1> { static const int value =  303; };
template <> struct vnode_shift<1, 1, 17,  2> { static const int value =  81; };
template <> struct vnode_shift<1, 1, 17,  3> { static const int value =  358; };
template <> struct vnode_shift<1, 1, 17,  4> { static const int value =  375; };
template <> struct vnode_shift<1, 1, 17,  5> { static const int value =  0; };
// ILS = 1, CHECK_NODE = 18
template <> struct vnode_shift<1, 1, 18,  0> { static const int value =  130; };
template <> struct vnode_shift<1, 1, 18,  1> { static const int value =  163; };
template <> struct vnode_shift<1, 1, 18,  2> { static const int value =  280; };
template <> struct vnode_shift<1, 1, 18,  3> { static const int value =  132; };
template <> struct vnode_shift<1, 1, 18,  4> { static const int value =  4; };
template <> struct vnode_shift<1, 1, 18,  5> { static const int value =  0; };
// ILS = 1, CHECK_NODE = 19
template <> struct vnode_shift<1, 1, 19,  0> { static const int value =  145; };
template <> struct vnode_shift<1, 1, 19,  1> { static const int value =  213; };
template <> struct vnode_shift<1, 1, 19,  2> { static const int value =  344; };
template <> struct vnode_shift<1, 1, 19,  3> { static const int value =  242; };
template <> struct vnode_shift<1, 1, 19,  4> { static const int value =  197; };
template <> struct vnode_shift<1, 1, 19,  5> { static const int value =  0; };
// ILS = 1, CHECK_NODE = 20
template <> struct vnode_shift<1, 1, 20,  0> { static const int value =  187; };
template <> struct vnode_shift<1, 1, 20,  1> { static const int value =  206; };
template <> struct vnode_shift<1, 1, 20,  2> { static const int value =  264; };
template <> struct vnode_shift<1, 1, 20,  3> { static const int value =  341; };
template <> struct vnode_shift<1, 1, 20,  4> { static const int value =  59; };
template <> struct vnode_shift<1, 1, 20,  5> { static const int value =  0; };
// ILS = 1, CHECK_NODE = 21
template <> struct vnode_shift<1, 1, 21,  0> { static const int value =  205; };
template <> struct vnode_shift<1, 1, 21,  1> { static const int value =  102; };
template <> struct vnode_shift<1, 1, 21,  2> { static const int value =  328; };
template <> struct vnode_shift<1, 1, 21,  3> { static const int value =  213; };
template <> struct vnode_shift<1, 1, 21,  4> { static const int value =  97; };
template <> struct vnode_shift<1, 1, 21,  5> { static const int value =  0; };
// ILS = 1, CHECK_NODE = 22
template <> struct vnode_shift<1, 1, 22,  0> { static const int value =  30; };
template <> struct vnode_shift<1, 1, 22,  1> { static const int value =  11; };
template <> struct vnode_shift<1, 1, 22,  2> { static const int value =  233; };
template <> struct vnode_shift<1, 1, 22,  3> { static const int value =  22; };
template <> struct vnode_shift<1, 1, 22,  4> { static const int value =  0; };
// ILS = 1, CHECK_NODE = 23
template <> struct vnode_shift<1, 1, 23,  0> { static const int value =  24; };
template <> struct vnode_shift<1, 1, 23,  1> { static const int value =  89; };
template <> struct vnode_shift<1, 1, 23,  2> { static const int value =  61; };
template <> struct vnode_shift<1, 1, 23,  3> { static const int value =  27; };
template <> struct vnode_shift<1, 1, 23,  4> { static const int value =  0; };
// ILS = 1, CHECK_NODE = 24
template <> struct vnode_shift<1, 1, 24,  0> { static const int value =  298; };
template <> struct vnode_shift<1, 1, 24,  1> { static const int value =  158; };
template <> struct vnode_shift<1, 1, 24,  2> { static const int value =  235; };
template <> struct vnode_shift<1, 1, 24,  3> { static const int value =  339; };
template <> struct vnode_shift<1, 1, 24,  4> { static const int value =  234; };
template <> struct vnode_shift<1, 1, 24,  5> { static const int value =  0; };
// ILS = 1, CHECK_NODE = 25
template <> struct vnode_shift<1, 1, 25,  0> { static const int value =  72; };
template <> struct vnode_shift<1, 1, 25,  1> { static const int value =  17; };
template <> struct vnode_shift<1, 1, 25,  2> { static const int value =  383; };
template <> struct vnode_shift<1, 1, 25,  3> { static const int value =  312; };
template <> struct vnode_shift<1, 1, 25,  4> { static const int value =  0; };
// ILS = 1, CHECK_NODE = 26
template <> struct vnode_shift<1, 1, 26,  0> { static const int value =  71; };
template <> struct vnode_shift<1, 1, 26,  1> { static const int value =  81; };
template <> struct vnode_shift<1, 1, 26,  2> { static const int value =  76; };
template <> struct vnode_shift<1, 1, 26,  3> { static const int value =  136; };
template <> struct vnode_shift<1, 1, 26,  4> { static const int value =  0; };
// ILS = 1, CHECK_NODE = 27
template <> struct vnode_shift<1, 1, 27,  0> { static const int value =  194; };
template <> struct vnode_shift<1, 1, 27,  1> { static const int value =  194; };
template <> struct vnode_shift<1, 1, 27,  2> { static const int value =  101; };
template <> struct vnode_shift<1, 1, 27,  3> { static const int value =  0; };
// ILS = 1, CHECK_NODE = 28
template <> struct vnode_shift<1, 1, 28,  0> { static const int value =  222; };
template <> struct vnode_shift<1, 1, 28,  1> { static const int value =  19; };
template <> struct vnode_shift<1, 1, 28,  2> { static const int value =  244; };
template <> struct vnode_shift<1, 1, 28,  3> { static const int value =  274; };
template <> struct vnode_shift<1, 1, 28,  4> { static const int value =  0; };
// ILS = 1, CHECK_NODE = 29
template <> struct vnode_shift<1, 1, 29,  0> { static const int value =  252; };
template <> struct vnode_shift<1, 1, 29,  1> { static const int value =  5; };
template <> struct vnode_shift<1, 1, 29,  2> { static const int value =  147; };
template <> struct vnode_shift<1, 1, 29,  3> { static const int value =  78; };
template <> struct vnode_shift<1, 1, 29,  4> { static const int value =  0; };
// ILS = 1, CHECK_NODE = 30
template <> struct vnode_shift<1, 1, 30,  0> { static const int value =  159; };
template <> struct vnode_shift<1, 1, 30,  1> { static const int value =  229; };
template <> struct vnode_shift<1, 1, 30,  2> { static const int value =  260; };
template <> struct vnode_shift<1, 1, 30,  3> { static const int value =  90; };
template <> struct vnode_shift<1, 1, 30,  4> { static const int value =  0; };
// ILS = 1, CHECK_NODE = 31
template <> struct vnode_shift<1, 1, 31,  0> { static const int value =  100; };
template <> struct vnode_shift<1, 1, 31,  1> { static const int value =  215; };
template <> struct vnode_shift<1, 1, 31,  2> { static const int value =  258; };
template <> struct vnode_shift<1, 1, 31,  3> { static const int value =  256; };
template <> struct vnode_shift<1, 1, 31,  4> { static const int value =  0; };
// ILS = 1, CHECK_NODE = 32
template <> struct vnode_shift<1, 1, 32,  0> { static const int value =  102; };
template <> struct vnode_shift<1, 1, 32,  1> { static const int value =  201; };
template <> struct vnode_shift<1, 1, 32,  2> { static const int value =  175; };
template <> struct vnode_shift<1, 1, 32,  3> { static const int value =  287; };
template <> struct vnode_shift<1, 1, 32,  4> { static const int value =  0; };
// ILS = 1, CHECK_NODE = 33
template <> struct vnode_shift<1, 1, 33,  0> { static const int value =  323; };
template <> struct vnode_shift<1, 1, 33,  1> { static const int value =  8; };
template <> struct vnode_shift<1, 1, 33,  2> { static const int value =  361; };
template <> struct vnode_shift<1, 1, 33,  3> { static const int value =  105; };
template <> struct vnode_shift<1, 1, 33,  4> { static const int value =  0; };
// ILS = 1, CHECK_NODE = 34
template <> struct vnode_shift<1, 1, 34,  0> { static const int value =  230; };
template <> struct vnode_shift<1, 1, 34,  1> { static const int value =  148; };
template <> struct vnode_shift<1, 1, 34,  2> { static const int value =  202; };
template <> struct vnode_shift<1, 1, 34,  3> { static const int value =  312; };
template <> struct vnode_shift<1, 1, 34,  4> { static const int value =  0; };
// ILS = 1, CHECK_NODE = 35
template <> struct vnode_shift<1, 1, 35,  0> { static const int value =  320; };
template <> struct vnode_shift<1, 1, 35,  1> { static const int value =  335; };
template <> struct vnode_shift<1, 1, 35,  2> { static const int value =  2; };
template <> struct vnode_shift<1, 1, 35,  3> { static const int value =  266; };
template <> struct vnode_shift<1, 1, 35,  4> { static const int value =  0; };
// ILS = 1, CHECK_NODE = 36
template <> struct vnode_shift<1, 1, 36,  0> { static const int value =  210; };
template <> struct vnode_shift<1, 1, 36,  1> { static const int value =  313; };
template <> struct vnode_shift<1, 1, 36,  2> { static const int value =  297; };
template <> struct vnode_shift<1, 1, 36,  3> { static const int value =  21; };
template <> struct vnode_shift<1, 1, 36,  4> { static const int value =  0; };
// ILS = 1, CHECK_NODE = 37
template <> struct vnode_shift<1, 1, 37,  0> { static const int value =  269; };
template <> struct vnode_shift<1, 1, 37,  1> { static const int value =  82; };
template <> struct vnode_shift<1, 1, 37,  2> { static const int value =  115; };
template <> struct vnode_shift<1, 1, 37,  3> { static const int value =  0; };
// ILS = 1, CHECK_NODE = 38
template <> struct vnode_shift<1, 1, 38,  0> { static const int value =  185; };
template <> struct vnode_shift<1, 1, 38,  1> { static const int value =  177; };
template <> struct vnode_shift<1, 1, 38,  2> { static const int value =  289; };
template <> struct vnode_shift<1, 1, 38,  3> { static const int value =  214; };
template <> struct vnode_shift<1, 1, 38,  4> { static const int value =  0; };
// ILS = 1, CHECK_NODE = 39
template <> struct vnode_shift<1, 1, 39,  0> { static const int value =  258; };
template <> struct vnode_shift<1, 1, 39,  1> { static const int value =  93; };
template <> struct vnode_shift<1, 1, 39,  2> { static const int value =  346; };
template <> struct vnode_shift<1, 1, 39,  3> { static const int value =  297; };
template <> struct vnode_shift<1, 1, 39,  4> { static const int value =  0; };
// ILS = 1, CHECK_NODE = 40
template <> struct vnode_shift<1, 1, 40,  0> { static const int value =  175; };
template <> struct vnode_shift<1, 1, 40,  1> { static const int value =  37; };
template <> struct vnode_shift<1, 1, 40,  2> { static const int value =  312; };
template <> struct vnode_shift<1, 1, 40,  3> { static const int value =  0; };
// ILS = 1, CHECK_NODE = 41
template <> struct vnode_shift<1, 1, 41,  0> { static const int value =  52; };
template <> struct vnode_shift<1, 1, 41,  1> { static const int value =  314; };
template <> struct vnode_shift<1, 1, 41,  2> { static const int value =  139; };
template <> struct vnode_shift<1, 1, 41,  3> { static const int value =  288; };
template <> struct vnode_shift<1, 1, 41,  4> { static const int value =  0; };
// ILS = 1, CHECK_NODE = 42
template <> struct vnode_shift<1, 1, 42,  0> { static const int value =  113; };
template <> struct vnode_shift<1, 1, 42,  1> { static const int value =  14; };
template <> struct vnode_shift<1, 1, 42,  2> { static const int value =  218; };
template <> struct vnode_shift<1, 1, 42,  3> { static const int value =  0; };
// ILS = 1, CHECK_NODE = 43
template <> struct vnode_shift<1, 1, 43,  0> { static const int value =  113; };
template <> struct vnode_shift<1, 1, 43,  1> { static const int value =  132; };
template <> struct vnode_shift<1, 1, 43,  2> { static const int value =  114; };
template <> struct vnode_shift<1, 1, 43,  3> { static const int value =  168; };
template <> struct vnode_shift<1, 1, 43,  4> { static const int value =  0; };
// ILS = 1, CHECK_NODE = 44
template <> struct vnode_shift<1, 1, 44,  0> { static const int value =  80; };
template <> struct vnode_shift<1, 1, 44,  1> { static const int value =  78; };
template <> struct vnode_shift<1, 1, 44,  2> { static const int value =  163; };
template <> struct vnode_shift<1, 1, 44,  3> { static const int value =  274; };
template <> struct vnode_shift<1, 1, 44,  4> { static const int value =  0; };
// ILS = 1, CHECK_NODE = 45
template <> struct vnode_shift<1, 1, 45,  0> { static const int value =  135; };
template <> struct vnode_shift<1, 1, 45,  1> { static const int value =  149; };
template <> struct vnode_shift<1, 1, 45,  2> { static const int value =  15; };
template <> struct vnode_shift<1, 1, 45,  3> { static const int value =  0; };
// ILS = 2, CHECK_NODE = 0
template <> struct vnode_shift<1, 2, 0,  0> { static const int value =  73; };
template <> struct vnode_shift<1, 2, 0,  1> { static const int value =  15; };
template <> struct vnode_shift<1, 2, 0,  2> { static const int value =  103; };
template <> struct vnode_shift<1, 2, 0,  3> { static const int value =  49; };
template <> struct vnode_shift<1, 2, 0,  4> { static const int value =  240; };
template <> struct vnode_shift<1, 2, 0,  5> { static const int value =  39; };
template <> struct vnode_shift<1, 2, 0,  6> { static const int value =  15; };
template <> struct vnode_shift<1, 2, 0,  7> { static const int value =  162; };
template <> struct vnode_shift<1, 2, 0,  8> { static const int value =  215; };
template <> struct vnode_shift<1, 2, 0,  9> { static const int value =  164; };
template <> struct vnode_shift<1, 2, 0,  10> { static const int value =  133; };
template <> struct vnode_shift<1, 2, 0,  11> { static const int value =  298; };
template <> struct vnode_shift<1, 2, 0,  12> { static const int value =  110; };
template <> struct vnode_shift<1, 2, 0,  13> { static const int value =  113; };
template <> struct vnode_shift<1, 2, 0,  14> { static const int value =  16; };
template <> struct vnode_shift<1, 2, 0,  15> { static const int value =  189; };
template <> struct vnode_shift<1, 2, 0,  16> { static const int value =  32; };
template <> struct vnode_shift<1, 2, 0,  17> { static const int value =  1; };
template <> struct vnode_shift<1, 2, 0,  18> { static const int value =  0; };
// ILS = 2, CHECK_NODE = 1
template <> struct vnode_shift<1, 2, 1,  0> { static const int value =  303; };
template <> struct vnode_shift<1, 2, 1,  1> { static const int value =  294; };
template <> struct vnode_shift<1, 2, 1,  2> { static const int value =  27; };
template <> struct vnode_shift<1, 2, 1,  3> { static const int value =  261; };
template <> struct vnode_shift<1, 2, 1,  4> { static const int value =  161; };
template <> struct vnode_shift<1, 2, 1,  5> { static const int value =  133; };
template <> struct vnode_shift<1, 2, 1,  6> { static const int value =  4; };
template <> struct vnode_shift<1, 2, 1,  7> { static const int value =  80; };
template <> struct vnode_shift<1, 2, 1,  8> { static const int value =  129; };
template <> struct vnode_shift<1, 2, 1,  9> { static const int value =  300; };
template <> struct vnode_shift<1, 2, 1,  10> { static const int value =  76; };
template <> struct vnode_shift<1, 2, 1,  11> { static const int value =  266; };
template <> struct vnode_shift<1, 2, 1,  12> { static const int value =  72; };
template <> struct vnode_shift<1, 2, 1,  13> { static const int value =  83; };
template <> struct vnode_shift<1, 2, 1,  14> { static const int value =  260; };
template <> struct vnode_shift<1, 2, 1,  15> { static const int value =  301; };
template <> struct vnode_shift<1, 2, 1,  16> { static const int value =  0; };
template <> struct vnode_shift<1, 2, 1,  17> { static const int value =  0; };
template <> struct vnode_shift<1, 2, 1,  18> { static const int value =  0; };
// ILS = 2, CHECK_NODE = 2
template <> struct vnode_shift<1, 2, 2,  0> { static const int value =  68; };
template <> struct vnode_shift<1, 2, 2,  1> { static const int value =  7; };
template <> struct vnode_shift<1, 2, 2,  2> { static const int value =  80; };
template <> struct vnode_shift<1, 2, 2,  3> { static const int value =  280; };
template <> struct vnode_shift<1, 2, 2,  4> { static const int value =  38; };
template <> struct vnode_shift<1, 2, 2,  5> { static const int value =  227; };
template <> struct vnode_shift<1, 2, 2,  6> { static const int value =  202; };
template <> struct vnode_shift<1, 2, 2,  7> { static const int value =  200; };
template <> struct vnode_shift<1, 2, 2,  8> { static const int value =  71; };
template <> struct vnode_shift<1, 2, 2,  9> { static const int value =  106; };
template <> struct vnode_shift<1, 2, 2,  10> { static const int value =  295; };
template <> struct vnode_shift<1, 2, 2,  11> { static const int value =  283; };
template <> struct vnode_shift<1, 2, 2,  12> { static const int value =  301; };
template <> struct vnode_shift<1, 2, 2,  13> { static const int value =  184; };
template <> struct vnode_shift<1, 2, 2,  14> { static const int value =  246; };
template <> struct vnode_shift<1, 2, 2,  15> { static const int value =  230; };
template <> struct vnode_shift<1, 2, 2,  16> { static const int value =  276; };
template <> struct vnode_shift<1, 2, 2,  17> { static const int value =  0; };
template <> struct vnode_shift<1, 2, 2,  18> { static const int value =  0; };
// ILS = 2, CHECK_NODE = 3
template <> struct vnode_shift<1, 2, 3,  0> { static const int value =  220; };
template <> struct vnode_shift<1, 2, 3,  1> { static const int value =  208; };
template <> struct vnode_shift<1, 2, 3,  2> { static const int value =  30; };
template <> struct vnode_shift<1, 2, 3,  3> { static const int value =  197; };
template <> struct vnode_shift<1, 2, 3,  4> { static const int value =  61; };
template <> struct vnode_shift<1, 2, 3,  5> { static const int value =  175; };
template <> struct vnode_shift<1, 2, 3,  6> { static const int value =  79; };
template <> struct vnode_shift<1, 2, 3,  7> { static const int value =  281; };
template <> struct vnode_shift<1, 2, 3,  8> { static const int value =  303; };
template <> struct vnode_shift<1, 2, 3,  9> { static const int value =  253; };
template <> struct vnode_shift<1, 2, 3,  10> { static const int value =  164; };
template <> struct vnode_shift<1, 2, 3,  11> { static const int value =  53; };
template <> struct vnode_shift<1, 2, 3,  12> { static const int value =  44; };
template <> struct vnode_shift<1, 2, 3,  13> { static const int value =  28; };
template <> struct vnode_shift<1, 2, 3,  14> { static const int value =  77; };
template <> struct vnode_shift<1, 2, 3,  15> { static const int value =  319; };
template <> struct vnode_shift<1, 2, 3,  16> { static const int value =  68; };
template <> struct vnode_shift<1, 2, 3,  17> { static const int value =  1; };
template <> struct vnode_shift<1, 2, 3,  18> { static const int value =  0; };
// ILS = 2, CHECK_NODE = 4
template <> struct vnode_shift<1, 2, 4,  0> { static const int value =  233; };
template <> struct vnode_shift<1, 2, 4,  1> { static const int value =  205; };
template <> struct vnode_shift<1, 2, 4,  2> { static const int value =  0; };
// ILS = 2, CHECK_NODE = 5
template <> struct vnode_shift<1, 2, 5,  0> { static const int value =  83; };
template <> struct vnode_shift<1, 2, 5,  1> { static const int value =  292; };
template <> struct vnode_shift<1, 2, 5,  2> { static const int value =  50; };
template <> struct vnode_shift<1, 2, 5,  3> { static const int value =  318; };
template <> struct vnode_shift<1, 2, 5,  4> { static const int value =  201; };
template <> struct vnode_shift<1, 2, 5,  5> { static const int value =  267; };
template <> struct vnode_shift<1, 2, 5,  6> { static const int value =  279; };
template <> struct vnode_shift<1, 2, 5,  7> { static const int value =  0; };
// ILS = 2, CHECK_NODE = 6
template <> struct vnode_shift<1, 2, 6,  0> { static const int value =  289; };
template <> struct vnode_shift<1, 2, 6,  1> { static const int value =  21; };
template <> struct vnode_shift<1, 2, 6,  2> { static const int value =  293; };
template <> struct vnode_shift<1, 2, 6,  3> { static const int value =  13; };
template <> struct vnode_shift<1, 2, 6,  4> { static const int value =  232; };
template <> struct vnode_shift<1, 2, 6,  5> { static const int value =  302; };
template <> struct vnode_shift<1, 2, 6,  6> { static const int value =  138; };
template <> struct vnode_shift<1, 2, 6,  7> { static const int value =  235; };
template <> struct vnode_shift<1, 2, 6,  8> { static const int value =  0; };
// ILS = 2, CHECK_NODE = 7
template <> struct vnode_shift<1, 2, 7,  0> { static const int value =  12; };
template <> struct vnode_shift<1, 2, 7,  1> { static const int value =  88; };
template <> struct vnode_shift<1, 2, 7,  2> { static const int value =  207; };
template <> struct vnode_shift<1, 2, 7,  3> { static const int value =  50; };
template <> struct vnode_shift<1, 2, 7,  4> { static const int value =  25; };
template <> struct vnode_shift<1, 2, 7,  5> { static const int value =  76; };
template <> struct vnode_shift<1, 2, 7,  6> { static const int value =  0; };
// ILS = 2, CHECK_NODE = 8
template <> struct vnode_shift<1, 2, 8,  0> { static const int value =  295; };
template <> struct vnode_shift<1, 2, 8,  1> { static const int value =  133; };
template <> struct vnode_shift<1, 2, 8,  2> { static const int value =  130; };
template <> struct vnode_shift<1, 2, 8,  3> { static const int value =  231; };
template <> struct vnode_shift<1, 2, 8,  4> { static const int value =  296; };
template <> struct vnode_shift<1, 2, 8,  5> { static const int value =  110; };
template <> struct vnode_shift<1, 2, 8,  6> { static const int value =  269; };
template <> struct vnode_shift<1, 2, 8,  7> { static const int value =  245; };
template <> struct vnode_shift<1, 2, 8,  8> { static const int value =  154; };
template <> struct vnode_shift<1, 2, 8,  9> { static const int value =  0; };
// ILS = 2, CHECK_NODE = 9
template <> struct vnode_shift<1, 2, 9,  0> { static const int value =  189; };
template <> struct vnode_shift<1, 2, 9,  1> { static const int value =  244; };
template <> struct vnode_shift<1, 2, 9,  2> { static const int value =  36; };
template <> struct vnode_shift<1, 2, 9,  3> { static const int value =  286; };
template <> struct vnode_shift<1, 2, 9,  4> { static const int value =  151; };
template <> struct vnode_shift<1, 2, 9,  5> { static const int value =  267; };
template <> struct vnode_shift<1, 2, 9,  6> { static const int value =  135; };
template <> struct vnode_shift<1, 2, 9,  7> { static const int value =  209; };
template <> struct vnode_shift<1, 2, 9,  8> { static const int value =  0; };
// ILS = 2, CHECK_NODE = 10
template <> struct vnode_shift<1, 2, 10,  0> { static const int value =  14; };
template <> struct vnode_shift<1, 2, 10,  1> { static const int value =  80; };
template <> struct vnode_shift<1, 2, 10,  2> { static const int value =  211; };
template <> struct vnode_shift<1, 2, 10,  3> { static const int value =  75; };
template <> struct vnode_shift<1, 2, 10,  4> { static const int value =  161; };
template <> struct vnode_shift<1, 2, 10,  5> { static const int value =  311; };
template <> struct vnode_shift<1, 2, 10,  6> { static const int value =  0; };
// ILS = 2, CHECK_NODE = 11
template <> struct vnode_shift<1, 2, 11,  0> { static const int value =  16; };
template <> struct vnode_shift<1, 2, 11,  1> { static const int value =  147; };
template <> struct vnode_shift<1, 2, 11,  2> { static const int value =  290; };
template <> struct vnode_shift<1, 2, 11,  3> { static const int value =  289; };
template <> struct vnode_shift<1, 2, 11,  4> { static const int value =  177; };
template <> struct vnode_shift<1, 2, 11,  5> { static const int value =  43; };
template <> struct vnode_shift<1, 2, 11,  6> { static const int value =  280; };
template <> struct vnode_shift<1, 2, 11,  7> { static const int value =  0; };
// ILS = 2, CHECK_NODE = 12
template <> struct vnode_shift<1, 2, 12,  0> { static const int value =  229; };
template <> struct vnode_shift<1, 2, 12,  1> { static const int value =  235; };
template <> struct vnode_shift<1, 2, 12,  2> { static const int value =  169; };
template <> struct vnode_shift<1, 2, 12,  3> { static const int value =  48; };
template <> struct vnode_shift<1, 2, 12,  4> { static const int value =  105; };
template <> struct vnode_shift<1, 2, 12,  5> { static const int value =  52; };
template <> struct vnode_shift<1, 2, 12,  6> { static const int value =  0; };
// ILS = 2, CHECK_NODE = 13
template <> struct vnode_shift<1, 2, 13,  0> { static const int value =  39; };
template <> struct vnode_shift<1, 2, 13,  1> { static const int value =  302; };
template <> struct vnode_shift<1, 2, 13,  2> { static const int value =  303; };
template <> struct vnode_shift<1, 2, 13,  3> { static const int value =  160; };
template <> struct vnode_shift<1, 2, 13,  4> { static const int value =  37; };
template <> struct vnode_shift<1, 2, 13,  5> { static const int value =  0; };
// ILS = 2, CHECK_NODE = 14
template <> struct vnode_shift<1, 2, 14,  0> { static const int value =  78; };
template <> struct vnode_shift<1, 2, 14,  1> { static const int value =  299; };
template <> struct vnode_shift<1, 2, 14,  2> { static const int value =  54; };
template <> struct vnode_shift<1, 2, 14,  3> { static const int value =  61; };
template <> struct vnode_shift<1, 2, 14,  4> { static const int value =  179; };
template <> struct vnode_shift<1, 2, 14,  5> { static const int value =  258; };
template <> struct vnode_shift<1, 2, 14,  6> { static const int value =  0; };
// ILS = 2, CHECK_NODE = 15
template <> struct vnode_shift<1, 2, 15,  0> { static const int value =  229; };
template <> struct vnode_shift<1, 2, 15,  1> { static const int value =  290; };
template <> struct vnode_shift<1, 2, 15,  2> { static const int value =  60; };
template <> struct vnode_shift<1, 2, 15,  3> { static const int value =  130; };
template <> struct vnode_shift<1, 2, 15,  4> { static const int value =  184; };
template <> struct vnode_shift<1, 2, 15,  5> { static const int value =  51; };
template <> struct vnode_shift<1, 2, 15,  6> { static const int value =  0; };
// ILS = 2, CHECK_NODE = 16
template <> struct vnode_shift<1, 2, 16,  0> { static const int value =  69; };
template <> struct vnode_shift<1, 2, 16,  1> { static const int value =  140; };
template <> struct vnode_shift<1, 2, 16,  2> { static const int value =  45; };
template <> struct vnode_shift<1, 2, 16,  3> { static const int value =  115; };
template <> struct vnode_shift<1, 2, 16,  4> { static const int value =  300; };
template <> struct vnode_shift<1, 2, 16,  5> { static const int value =  0; };
// ILS = 2, CHECK_NODE = 17
template <> struct vnode_shift<1, 2, 17,  0> { static const int value =  257; };
template <> struct vnode_shift<1, 2, 17,  1> { static const int value =  147; };
template <> struct vnode_shift<1, 2, 17,  2> { static const int value =  128; };
template <> struct vnode_shift<1, 2, 17,  3> { static const int value =  51; };
template <> struct vnode_shift<1, 2, 17,  4> { static const int value =  228; };
template <> struct vnode_shift<1, 2, 17,  5> { static const int value =  0; };
// ILS = 2, CHECK_NODE = 18
template <> struct vnode_shift<1, 2, 18,  0> { static const int value =  260; };
template <> struct vnode_shift<1, 2, 18,  1> { static const int value =  294; };
template <> struct vnode_shift<1, 2, 18,  2> { static const int value =  291; };
template <> struct vnode_shift<1, 2, 18,  3> { static const int value =  141; };
template <> struct vnode_shift<1, 2, 18,  4> { static const int value =  295; };
template <> struct vnode_shift<1, 2, 18,  5> { static const int value =  0; };
// ILS = 2, CHECK_NODE = 19
template <> struct vnode_shift<1, 2, 19,  0> { static const int value =  64; };
template <> struct vnode_shift<1, 2, 19,  1> { static const int value =  181; };
template <> struct vnode_shift<1, 2, 19,  2> { static const int value =  101; };
template <> struct vnode_shift<1, 2, 19,  3> { static const int value =  270; };
template <> struct vnode_shift<1, 2, 19,  4> { static const int value =  41; };
template <> struct vnode_shift<1, 2, 19,  5> { static const int value =  0; };
// ILS = 2, CHECK_NODE = 20
template <> struct vnode_shift<1, 2, 20,  0> { static const int value =  301; };
template <> struct vnode_shift<1, 2, 20,  1> { static const int value =  162; };
template <> struct vnode_shift<1, 2, 20,  2> { static const int value =  40; };
template <> struct vnode_shift<1, 2, 20,  3> { static const int value =  130; };
template <> struct vnode_shift<1, 2, 20,  4> { static const int value =  10; };
template <> struct vnode_shift<1, 2, 20,  5> { static const int value =  0; };
// ILS = 2, CHECK_NODE = 21
template <> struct vnode_shift<1, 2, 21,  0> { static const int value =  79; };
template <> struct vnode_shift<1, 2, 21,  1> { static const int value =  175; };
template <> struct vnode_shift<1, 2, 21,  2> { static const int value =  132; };
template <> struct vnode_shift<1, 2, 21,  3> { static const int value =  283; };
template <> struct vnode_shift<1, 2, 21,  4> { static const int value =  103; };
template <> struct vnode_shift<1, 2, 21,  5> { static const int value =  0; };
// ILS = 2, CHECK_NODE = 22
template <> struct vnode_shift<1, 2, 22,  0> { static const int value =  177; };
template <> struct vnode_shift<1, 2, 22,  1> { static const int value =  20; };
template <> struct vnode_shift<1, 2, 22,  2> { static const int value =  55; };
template <> struct vnode_shift<1, 2, 22,  3> { static const int value =  316; };
template <> struct vnode_shift<1, 2, 22,  4> { static const int value =  0; };
// ILS = 2, CHECK_NODE = 23
template <> struct vnode_shift<1, 2, 23,  0> { static const int value =  249; };
template <> struct vnode_shift<1, 2, 23,  1> { static const int value =  50; };
template <> struct vnode_shift<1, 2, 23,  2> { static const int value =  133; };
template <> struct vnode_shift<1, 2, 23,  3> { static const int value =  105; };
template <> struct vnode_shift<1, 2, 23,  4> { static const int value =  0; };
// ILS = 2, CHECK_NODE = 24
template <> struct vnode_shift<1, 2, 24,  0> { static const int value =  289; };
template <> struct vnode_shift<1, 2, 24,  1> { static const int value =  280; };
template <> struct vnode_shift<1, 2, 24,  2> { static const int value =  110; };
template <> struct vnode_shift<1, 2, 24,  3> { static const int value =  187; };
template <> struct vnode_shift<1, 2, 24,  4> { static const int value =  281; };
template <> struct vnode_shift<1, 2, 24,  5> { static const int value =  0; };
// ILS = 2, CHECK_NODE = 25
template <> struct vnode_shift<1, 2, 25,  0> { static const int value =  172; };
template <> struct vnode_shift<1, 2, 25,  1> { static const int value =  295; };
template <> struct vnode_shift<1, 2, 25,  2> { static const int value =  96; };
template <> struct vnode_shift<1, 2, 25,  3> { static const int value =  46; };
template <> struct vnode_shift<1, 2, 25,  4> { static const int value =  0; };
// ILS = 2, CHECK_NODE = 26
template <> struct vnode_shift<1, 2, 26,  0> { static const int value =  270; };
template <> struct vnode_shift<1, 2, 26,  1> { static const int value =  110; };
template <> struct vnode_shift<1, 2, 26,  2> { static const int value =  318; };
template <> struct vnode_shift<1, 2, 26,  3> { static const int value =  67; };
template <> struct vnode_shift<1, 2, 26,  4> { static const int value =  0; };
// ILS = 2, CHECK_NODE = 27
template <> struct vnode_shift<1, 2, 27,  0> { static const int value =  210; };
template <> struct vnode_shift<1, 2, 27,  1> { static const int value =  29; };
template <> struct vnode_shift<1, 2, 27,  2> { static const int value =  304; };
template <> struct vnode_shift<1, 2, 27,  3> { static const int value =  0; };
// ILS = 2, CHECK_NODE = 28
template <> struct vnode_shift<1, 2, 28,  0> { static const int value =  11; };
template <> struct vnode_shift<1, 2, 28,  1> { static const int value =  293; };
template <> struct vnode_shift<1, 2, 28,  2> { static const int value =  50; };
template <> struct vnode_shift<1, 2, 28,  3> { static const int value =  234; };
template <> struct vnode_shift<1, 2, 28,  4> { static const int value =  0; };
// ILS = 2, CHECK_NODE = 29
template <> struct vnode_shift<1, 2, 29,  0> { static const int value =  27; };
template <> struct vnode_shift<1, 2, 29,  1> { static const int value =  308; };
template <> struct vnode_shift<1, 2, 29,  2> { static const int value =  117; };
template <> struct vnode_shift<1, 2, 29,  3> { static const int value =  29; };
template <> struct vnode_shift<1, 2, 29,  4> { static const int value =  0; };
// ILS = 2, CHECK_NODE = 30
template <> struct vnode_shift<1, 2, 30,  0> { static const int value =  91; };
template <> struct vnode_shift<1, 2, 30,  1> { static const int value =  23; };
template <> struct vnode_shift<1, 2, 30,  2> { static const int value =  105; };
template <> struct vnode_shift<1, 2, 30,  3> { static const int value =  135; };
template <> struct vnode_shift<1, 2, 30,  4> { static const int value =  0; };
// ILS = 2, CHECK_NODE = 31
template <> struct vnode_shift<1, 2, 31,  0> { static const int value =  222; };
template <> struct vnode_shift<1, 2, 31,  1> { static const int value =  308; };
template <> struct vnode_shift<1, 2, 31,  2> { static const int value =  66; };
template <> struct vnode_shift<1, 2, 31,  3> { static const int value =  162; };
template <> struct vnode_shift<1, 2, 31,  4> { static const int value =  0; };
// ILS = 2, CHECK_NODE = 32
template <> struct vnode_shift<1, 2, 32,  0> { static const int value =  210; };
template <> struct vnode_shift<1, 2, 32,  1> { static const int value =  22; };
template <> struct vnode_shift<1, 2, 32,  2> { static const int value =  271; };
template <> struct vnode_shift<1, 2, 32,  3> { static const int value =  217; };
template <> struct vnode_shift<1, 2, 32,  4> { static const int value =  0; };
// ILS = 2, CHECK_NODE = 33
template <> struct vnode_shift<1, 2, 33,  0> { static const int value =  170; };
template <> struct vnode_shift<1, 2, 33,  1> { static const int value =  20; };
template <> struct vnode_shift<1, 2, 33,  2> { static const int value =  140; };
template <> struct vnode_shift<1, 2, 33,  3> { static const int value =  33; };
template <> struct vnode_shift<1, 2, 33,  4> { static const int value =  0; };
// ILS = 2, CHECK_NODE = 34
template <> struct vnode_shift<1, 2, 34,  0> { static const int value =  187; };
template <> struct vnode_shift<1, 2, 34,  1> { static const int value =  296; };
template <> struct vnode_shift<1, 2, 34,  2> { static const int value =  5; };
template <> struct vnode_shift<1, 2, 34,  3> { static const int value =  44; };
template <> struct vnode_shift<1, 2, 34,  4> { static const int value =  0; };
// ILS = 2, CHECK_NODE = 35
template <> struct vnode_shift<1, 2, 35,  0> { static const int value =  207; };
template <> struct vnode_shift<1, 2, 35,  1> { static const int value =  158; };
template <> struct vnode_shift<1, 2, 35,  2> { static const int value =  55; };
template <> struct vnode_shift<1, 2, 35,  3> { static const int value =  285; };
template <> struct vnode_shift<1, 2, 35,  4> { static const int value =  0; };
// ILS = 2, CHECK_NODE = 36
template <> struct vnode_shift<1, 2, 36,  0> { static const int value =  259; };
template <> struct vnode_shift<1, 2, 36,  1> { static const int value =  179; };
template <> struct vnode_shift<1, 2, 36,  2> { static const int value =  178; };
template <> struct vnode_shift<1, 2, 36,  3> { static const int value =  160; };
template <> struct vnode_shift<1, 2, 36,  4> { static const int value =  0; };
// ILS = 2, CHECK_NODE = 37
template <> struct vnode_shift<1, 2, 37,  0> { static const int value =  298; };
template <> struct vnode_shift<1, 2, 37,  1> { static const int value =  15; };
template <> struct vnode_shift<1, 2, 37,  2> { static const int value =  115; };
template <> struct vnode_shift<1, 2, 37,  3> { static const int value =  0; };
// ILS = 2, CHECK_NODE = 38
template <> struct vnode_shift<1, 2, 38,  0> { static const int value =  151; };
template <> struct vnode_shift<1, 2, 38,  1> { static const int value =  179; };
template <> struct vnode_shift<1, 2, 38,  2> { static const int value =  64; };
template <> struct vnode_shift<1, 2, 38,  3> { static const int value =  181; };
template <> struct vnode_shift<1, 2, 38,  4> { static const int value =  0; };
// ILS = 2, CHECK_NODE = 39
template <> struct vnode_shift<1, 2, 39,  0> { static const int value =  102; };
template <> struct vnode_shift<1, 2, 39,  1> { static const int value =  77; };
template <> struct vnode_shift<1, 2, 39,  2> { static const int value =  192; };
template <> struct vnode_shift<1, 2, 39,  3> { static const int value =  208; };
template <> struct vnode_shift<1, 2, 39,  4> { static const int value =  0; };
// ILS = 2, CHECK_NODE = 40
template <> struct vnode_shift<1, 2, 40,  0> { static const int value =  32; };
template <> struct vnode_shift<1, 2, 40,  1> { static const int value =  80; };
template <> struct vnode_shift<1, 2, 40,  2> { static const int value =  197; };
template <> struct vnode_shift<1, 2, 40,  3> { static const int value =  0; };
// ILS = 2, CHECK_NODE = 41
template <> struct vnode_shift<1, 2, 41,  0> { static const int value =  154; };
template <> struct vnode_shift<1, 2, 41,  1> { static const int value =  47; };
template <> struct vnode_shift<1, 2, 41,  2> { static const int value =  124; };
template <> struct vnode_shift<1, 2, 41,  3> { static const int value =  207; };
template <> struct vnode_shift<1, 2, 41,  4> { static const int value =  0; };
// ILS = 2, CHECK_NODE = 42
template <> struct vnode_shift<1, 2, 42,  0> { static const int value =  226; };
template <> struct vnode_shift<1, 2, 42,  1> { static const int value =  65; };
template <> struct vnode_shift<1, 2, 42,  2> { static const int value =  126; };
template <> struct vnode_shift<1, 2, 42,  3> { static const int value =  0; };
// ILS = 2, CHECK_NODE = 43
template <> struct vnode_shift<1, 2, 43,  0> { static const int value =  228; };
template <> struct vnode_shift<1, 2, 43,  1> { static const int value =  69; };
template <> struct vnode_shift<1, 2, 43,  2> { static const int value =  176; };
template <> struct vnode_shift<1, 2, 43,  3> { static const int value =  102; };
template <> struct vnode_shift<1, 2, 43,  4> { static const int value =  0; };
// ILS = 2, CHECK_NODE = 44
template <> struct vnode_shift<1, 2, 44,  0> { static const int value =  234; };
template <> struct vnode_shift<1, 2, 44,  1> { static const int value =  227; };
template <> struct vnode_shift<1, 2, 44,  2> { static const int value =  259; };
template <> struct vnode_shift<1, 2, 44,  3> { static const int value =  260; };
template <> struct vnode_shift<1, 2, 44,  4> { static const int value =  0; };
// ILS = 2, CHECK_NODE = 45
template <> struct vnode_shift<1, 2, 45,  0> { static const int value =  101; };
template <> struct vnode_shift<1, 2, 45,  1> { static const int value =  228; };
template <> struct vnode_shift<1, 2, 45,  2> { static const int value =  126; };
template <> struct vnode_shift<1, 2, 45,  3> { static const int value =  0; };
// ILS = 3, CHECK_NODE = 0
template <> struct vnode_shift<1, 3, 0,  0> { static const int value =  223; };
template <> struct vnode_shift<1, 3, 0,  1> { static const int value =  16; };
template <> struct vnode_shift<1, 3, 0,  2> { static const int value =  94; };
template <> struct vnode_shift<1, 3, 0,  3> { static const int value =  91; };
template <> struct vnode_shift<1, 3, 0,  4> { static const int value =  74; };
template <> struct vnode_shift<1, 3, 0,  5> { static const int value =  10; };
template <> struct vnode_shift<1, 3, 0,  6> { static const int value =  0; };
template <> struct vnode_shift<1, 3, 0,  7> { static const int value =  205; };
template <> struct vnode_shift<1, 3, 0,  8> { static const int value =  216; };
template <> struct vnode_shift<1, 3, 0,  9> { static const int value =  21; };
template <> struct vnode_shift<1, 3, 0,  10> { static const int value =  215; };
template <> struct vnode_shift<1, 3, 0,  11> { static const int value =  14; };
template <> struct vnode_shift<1, 3, 0,  12> { static const int value =  70; };
template <> struct vnode_shift<1, 3, 0,  13> { static const int value =  141; };
template <> struct vnode_shift<1, 3, 0,  14> { static const int value =  198; };
template <> struct vnode_shift<1, 3, 0,  15> { static const int value =  104; };
template <> struct vnode_shift<1, 3, 0,  16> { static const int value =  81; };
template <> struct vnode_shift<1, 3, 0,  17> { static const int value =  1; };
template <> struct vnode_shift<1, 3, 0,  18> { static const int value =  0; };
// ILS = 3, CHECK_NODE = 1
template <> struct vnode_shift<1, 3, 1,  0> { static const int value =  141; };
template <> struct vnode_shift<1, 3, 1,  1> { static const int value =  45; };
template <> struct vnode_shift<1, 3, 1,  2> { static const int value =  151; };
template <> struct vnode_shift<1, 3, 1,  3> { static const int value =  46; };
template <> struct vnode_shift<1, 3, 1,  4> { static const int value =  119; };
template <> struct vnode_shift<1, 3, 1,  5> { static const int value =  157; };
template <> struct vnode_shift<1, 3, 1,  6> { static const int value =  133; };
template <> struct vnode_shift<1, 3, 1,  7> { static const int value =  87; };
template <> struct vnode_shift<1, 3, 1,  8> { static const int value =  206; };
template <> struct vnode_shift<1, 3, 1,  9> { static const int value =  93; };
template <> struct vnode_shift<1, 3, 1,  10> { static const int value =  79; };
template <> struct vnode_shift<1, 3, 1,  11> { static const int value =  9; };
template <> struct vnode_shift<1, 3, 1,  12> { static const int value =  118; };
template <> struct vnode_shift<1, 3, 1,  13> { static const int value =  194; };
template <> struct vnode_shift<1, 3, 1,  14> { static const int value =  31; };
template <> struct vnode_shift<1, 3, 1,  15> { static const int value =  187; };
template <> struct vnode_shift<1, 3, 1,  16> { static const int value =  0; };
template <> struct vnode_shift<1, 3, 1,  17> { static const int value =  0; };
template <> struct vnode_shift<1, 3, 1,  18> { static const int value =  0; };
// ILS = 3, CHECK_NODE = 2
template <> struct vnode_shift<1, 3, 2,  0> { static const int value =  207; };
template <> struct vnode_shift<1, 3, 2,  1> { static const int value =  203; };
template <> struct vnode_shift<1, 3, 2,  2> { static const int value =  31; };
template <> struct vnode_shift<1, 3, 2,  3> { static const int value =  176; };
template <> struct vnode_shift<1, 3, 2,  4> { static const int value =  180; };
template <> struct vnode_shift<1, 3, 2,  5> { static const int value =  186; };
template <> struct vnode_shift<1, 3, 2,  6> { static const int value =  95; };
template <> struct vnode_shift<1, 3, 2,  7> { static const int value =  153; };
template <> struct vnode_shift<1, 3, 2,  8> { static const int value =  177; };
template <> struct vnode_shift<1, 3, 2,  9> { static const int value =  70; };
template <> struct vnode_shift<1, 3, 2,  10> { static const int value =  77; };
template <> struct vnode_shift<1, 3, 2,  11> { static const int value =  214; };
template <> struct vnode_shift<1, 3, 2,  12> { static const int value =  77; };
template <> struct vnode_shift<1, 3, 2,  13> { static const int value =  198; };
template <> struct vnode_shift<1, 3, 2,  14> { static const int value =  117; };
template <> struct vnode_shift<1, 3, 2,  15> { static const int value =  223; };
template <> struct vnode_shift<1, 3, 2,  16> { static const int value =  90; };
template <> struct vnode_shift<1, 3, 2,  17> { static const int value =  0; };
template <> struct vnode_shift<1, 3, 2,  18> { static const int value =  0; };
// ILS = 3, CHECK_NODE = 3
template <> struct vnode_shift<1, 3, 3,  0> { static const int value =  201; };
template <> struct vnode_shift<1, 3, 3,  1> { static const int value =  18; };
template <> struct vnode_shift<1, 3, 3,  2> { static const int value =  165; };
template <> struct vnode_shift<1, 3, 3,  3> { static const int value =  5; };
template <> struct vnode_shift<1, 3, 3,  4> { static const int value =  45; };
template <> struct vnode_shift<1, 3, 3,  5> { static const int value =  142; };
template <> struct vnode_shift<1, 3, 3,  6> { static const int value =  16; };
template <> struct vnode_shift<1, 3, 3,  7> { static const int value =  34; };
template <> struct vnode_shift<1, 3, 3,  8> { static const int value =  155; };
template <> struct vnode_shift<1, 3, 3,  9> { static const int value =  213; };
template <> struct vnode_shift<1, 3, 3,  10> { static const int value =  147; };
template <> struct vnode_shift<1, 3, 3,  11> { static const int value =  69; };
template <> struct vnode_shift<1, 3, 3,  12> { static const int value =  96; };
template <> struct vnode_shift<1, 3, 3,  13> { static const int value =  74; };
template <> struct vnode_shift<1, 3, 3,  14> { static const int value =  99; };
template <> struct vnode_shift<1, 3, 3,  15> { static const int value =  30; };
template <> struct vnode_shift<1, 3, 3,  16> { static const int value =  158; };
template <> struct vnode_shift<1, 3, 3,  17> { static const int value =  1; };
template <> struct vnode_shift<1, 3, 3,  18> { static const int value =  0; };
// ILS = 3, CHECK_NODE = 4
template <> struct vnode_shift<1, 3, 4,  0> { static const int value =  170; };
template <> struct vnode_shift<1, 3, 4,  1> { static const int value =  10; };
template <> struct vnode_shift<1, 3, 4,  2> { static const int value =  0; };
// ILS = 3, CHECK_NODE = 5
template <> struct vnode_shift<1, 3, 5,  0> { static const int value =  164; };
template <> struct vnode_shift<1, 3, 5,  1> { static const int value =  59; };
template <> struct vnode_shift<1, 3, 5,  2> { static const int value =  86; };
template <> struct vnode_shift<1, 3, 5,  3> { static const int value =  80; };
template <> struct vnode_shift<1, 3, 5,  4> { static const int value =  182; };
template <> struct vnode_shift<1, 3, 5,  5> { static const int value =  130; };
template <> struct vnode_shift<1, 3, 5,  6> { static const int value =  153; };
template <> struct vnode_shift<1, 3, 5,  7> { static const int value =  0; };
// ILS = 3, CHECK_NODE = 6
template <> struct vnode_shift<1, 3, 6,  0> { static const int value =  158; };
template <> struct vnode_shift<1, 3, 6,  1> { static const int value =  119; };
template <> struct vnode_shift<1, 3, 6,  2> { static const int value =  113; };
template <> struct vnode_shift<1, 3, 6,  3> { static const int value =  21; };
template <> struct vnode_shift<1, 3, 6,  4> { static const int value =  63; };
template <> struct vnode_shift<1, 3, 6,  5> { static const int value =  51; };
template <> struct vnode_shift<1, 3, 6,  6> { static const int value =  136; };
template <> struct vnode_shift<1, 3, 6,  7> { static const int value =  116; };
template <> struct vnode_shift<1, 3, 6,  8> { static const int value =  0; };
// ILS = 3, CHECK_NODE = 7
template <> struct vnode_shift<1, 3, 7,  0> { static const int value =  17; };
template <> struct vnode_shift<1, 3, 7,  1> { static const int value =  76; };
template <> struct vnode_shift<1, 3, 7,  2> { static const int value =  104; };
template <> struct vnode_shift<1, 3, 7,  3> { static const int value =  100; };
template <> struct vnode_shift<1, 3, 7,  4> { static const int value =  150; };
template <> struct vnode_shift<1, 3, 7,  5> { static const int value =  158; };
template <> struct vnode_shift<1, 3, 7,  6> { static const int value =  0; };
// ILS = 3, CHECK_NODE = 8
template <> struct vnode_shift<1, 3, 8,  0> { static const int value =  33; };
template <> struct vnode_shift<1, 3, 8,  1> { static const int value =  95; };
template <> struct vnode_shift<1, 3, 8,  2> { static const int value =  4; };
template <> struct vnode_shift<1, 3, 8,  3> { static const int value =  217; };
template <> struct vnode_shift<1, 3, 8,  4> { static const int value =  204; };
template <> struct vnode_shift<1, 3, 8,  5> { static const int value =  39; };
template <> struct vnode_shift<1, 3, 8,  6> { static const int value =  58; };
template <> struct vnode_shift<1, 3, 8,  7> { static const int value =  44; };
template <> struct vnode_shift<1, 3, 8,  8> { static const int value =  201; };
template <> struct vnode_shift<1, 3, 8,  9> { static const int value =  0; };
// ILS = 3, CHECK_NODE = 9
template <> struct vnode_shift<1, 3, 9,  0> { static const int value =  9; };
template <> struct vnode_shift<1, 3, 9,  1> { static const int value =  37; };
template <> struct vnode_shift<1, 3, 9,  2> { static const int value =  213; };
template <> struct vnode_shift<1, 3, 9,  3> { static const int value =  105; };
template <> struct vnode_shift<1, 3, 9,  4> { static const int value =  89; };
template <> struct vnode_shift<1, 3, 9,  5> { static const int value =  185; };
template <> struct vnode_shift<1, 3, 9,  6> { static const int value =  109; };
template <> struct vnode_shift<1, 3, 9,  7> { static const int value =  218; };
template <> struct vnode_shift<1, 3, 9,  8> { static const int value =  0; };
// ILS = 3, CHECK_NODE = 10
template <> struct vnode_shift<1, 3, 10,  0> { static const int value =  82; };
template <> struct vnode_shift<1, 3, 10,  1> { static const int value =  165; };
template <> struct vnode_shift<1, 3, 10,  2> { static const int value =  174; };
template <> struct vnode_shift<1, 3, 10,  3> { static const int value =  19; };
template <> struct vnode_shift<1, 3, 10,  4> { static const int value =  194; };
template <> struct vnode_shift<1, 3, 10,  5> { static const int value =  103; };
template <> struct vnode_shift<1, 3, 10,  6> { static const int value =  0; };
// ILS = 3, CHECK_NODE = 11
template <> struct vnode_shift<1, 3, 11,  0> { static const int value =  52; };
template <> struct vnode_shift<1, 3, 11,  1> { static const int value =  11; };
template <> struct vnode_shift<1, 3, 11,  2> { static const int value =  2; };
template <> struct vnode_shift<1, 3, 11,  3> { static const int value =  35; };
template <> struct vnode_shift<1, 3, 11,  4> { static const int value =  32; };
template <> struct vnode_shift<1, 3, 11,  5> { static const int value =  84; };
template <> struct vnode_shift<1, 3, 11,  6> { static const int value =  201; };
template <> struct vnode_shift<1, 3, 11,  7> { static const int value =  0; };
// ILS = 3, CHECK_NODE = 12
template <> struct vnode_shift<1, 3, 12,  0> { static const int value =  142; };
template <> struct vnode_shift<1, 3, 12,  1> { static const int value =  175; };
template <> struct vnode_shift<1, 3, 12,  2> { static const int value =  136; };
template <> struct vnode_shift<1, 3, 12,  3> { static const int value =  3; };
template <> struct vnode_shift<1, 3, 12,  4> { static const int value =  28; };
template <> struct vnode_shift<1, 3, 12,  5> { static const int value =  182; };
template <> struct vnode_shift<1, 3, 12,  6> { static const int value =  0; };
// ILS = 3, CHECK_NODE = 13
template <> struct vnode_shift<1, 3, 13,  0> { static const int value =  81; };
template <> struct vnode_shift<1, 3, 13,  1> { static const int value =  56; };
template <> struct vnode_shift<1, 3, 13,  2> { static const int value =  72; };
template <> struct vnode_shift<1, 3, 13,  3> { static const int value =  217; };
template <> struct vnode_shift<1, 3, 13,  4> { static const int value =  78; };
template <> struct vnode_shift<1, 3, 13,  5> { static const int value =  0; };
// ILS = 3, CHECK_NODE = 14
template <> struct vnode_shift<1, 3, 14,  0> { static const int value =  14; };
template <> struct vnode_shift<1, 3, 14,  1> { static const int value =  175; };
template <> struct vnode_shift<1, 3, 14,  2> { static const int value =  211; };
template <> struct vnode_shift<1, 3, 14,  3> { static const int value =  191; };
template <> struct vnode_shift<1, 3, 14,  4> { static const int value =  51; };
template <> struct vnode_shift<1, 3, 14,  5> { static const int value =  43; };
template <> struct vnode_shift<1, 3, 14,  6> { static const int value =  0; };
// ILS = 3, CHECK_NODE = 15
template <> struct vnode_shift<1, 3, 15,  0> { static const int value =  90; };
template <> struct vnode_shift<1, 3, 15,  1> { static const int value =  120; };
template <> struct vnode_shift<1, 3, 15,  2> { static const int value =  131; };
template <> struct vnode_shift<1, 3, 15,  3> { static const int value =  209; };
template <> struct vnode_shift<1, 3, 15,  4> { static const int value =  209; };
template <> struct vnode_shift<1, 3, 15,  5> { static const int value =  81; };
template <> struct vnode_shift<1, 3, 15,  6> { static const int value =  0; };
// ILS = 3, CHECK_NODE = 16
template <> struct vnode_shift<1, 3, 16,  0> { static const int value =  154; };
template <> struct vnode_shift<1, 3, 16,  1> { static const int value =  164; };
template <> struct vnode_shift<1, 3, 16,  2> { static const int value =  43; };
template <> struct vnode_shift<1, 3, 16,  3> { static const int value =  189; };
template <> struct vnode_shift<1, 3, 16,  4> { static const int value =  101; };
template <> struct vnode_shift<1, 3, 16,  5> { static const int value =  0; };
// ILS = 3, CHECK_NODE = 17
template <> struct vnode_shift<1, 3, 17,  0> { static const int value =  56; };
template <> struct vnode_shift<1, 3, 17,  1> { static const int value =  110; };
template <> struct vnode_shift<1, 3, 17,  2> { static const int value =  200; };
template <> struct vnode_shift<1, 3, 17,  3> { static const int value =  63; };
template <> struct vnode_shift<1, 3, 17,  4> { static const int value =  4; };
template <> struct vnode_shift<1, 3, 17,  5> { static const int value =  0; };
// ILS = 3, CHECK_NODE = 18
template <> struct vnode_shift<1, 3, 18,  0> { static const int value =  199; };
template <> struct vnode_shift<1, 3, 18,  1> { static const int value =  110; };
template <> struct vnode_shift<1, 3, 18,  2> { static const int value =  200; };
template <> struct vnode_shift<1, 3, 18,  3> { static const int value =  143; };
template <> struct vnode_shift<1, 3, 18,  4> { static const int value =  186; };
template <> struct vnode_shift<1, 3, 18,  5> { static const int value =  0; };
// ILS = 3, CHECK_NODE = 19
template <> struct vnode_shift<1, 3, 19,  0> { static const int value =  8; };
template <> struct vnode_shift<1, 3, 19,  1> { static const int value =  6; };
template <> struct vnode_shift<1, 3, 19,  2> { static const int value =  103; };
template <> struct vnode_shift<1, 3, 19,  3> { static const int value =  198; };
template <> struct vnode_shift<1, 3, 19,  4> { static const int value =  8; };
template <> struct vnode_shift<1, 3, 19,  5> { static const int value =  0; };
// ILS = 3, CHECK_NODE = 20
template <> struct vnode_shift<1, 3, 20,  0> { static const int value =  105; };
template <> struct vnode_shift<1, 3, 20,  1> { static const int value =  210; };
template <> struct vnode_shift<1, 3, 20,  2> { static const int value =  121; };
template <> struct vnode_shift<1, 3, 20,  3> { static const int value =  214; };
template <> struct vnode_shift<1, 3, 20,  4> { static const int value =  183; };
template <> struct vnode_shift<1, 3, 20,  5> { static const int value =  0; };
// ILS = 3, CHECK_NODE = 21
template <> struct vnode_shift<1, 3, 21,  0> { static const int value =  192; };
template <> struct vnode_shift<1, 3, 21,  1> { static const int value =  131; };
template <> struct vnode_shift<1, 3, 21,  2> { static const int value =  220; };
template <> struct vnode_shift<1, 3, 21,  3> { static const int value =  50; };
template <> struct vnode_shift<1, 3, 21,  4> { static const int value =  106; };
template <> struct vnode_shift<1, 3, 21,  5> { static const int value =  0; };
// ILS = 3, CHECK_NODE = 22
template <> struct vnode_shift<1, 3, 22,  0> { static const int value =  53; };
template <> struct vnode_shift<1, 3, 22,  1> { static const int value =  0; };
template <> struct vnode_shift<1, 3, 22,  2> { static const int value =  3; };
template <> struct vnode_shift<1, 3, 22,  3> { static const int value =  148; };
template <> struct vnode_shift<1, 3, 22,  4> { static const int value =  0; };
// ILS = 3, CHECK_NODE = 23
template <> struct vnode_shift<1, 3, 23,  0> { static const int value =  88; };
template <> struct vnode_shift<1, 3, 23,  1> { static const int value =  203; };
template <> struct vnode_shift<1, 3, 23,  2> { static const int value =  168; };
template <> struct vnode_shift<1, 3, 23,  3> { static const int value =  122; };
template <> struct vnode_shift<1, 3, 23,  4> { static const int value =  0; };
// ILS = 3, CHECK_NODE = 24
template <> struct vnode_shift<1, 3, 24,  0> { static const int value =  49; };
template <> struct vnode_shift<1, 3, 24,  1> { static const int value =  157; };
template <> struct vnode_shift<1, 3, 24,  2> { static const int value =  64; };
template <> struct vnode_shift<1, 3, 24,  3> { static const int value =  193; };
template <> struct vnode_shift<1, 3, 24,  4> { static const int value =  124; };
template <> struct vnode_shift<1, 3, 24,  5> { static const int value =  0; };
// ILS = 3, CHECK_NODE = 25
template <> struct vnode_shift<1, 3, 25,  0> { static const int value =  1; };
template <> struct vnode_shift<1, 3, 25,  1> { static const int value =  166; };
template <> struct vnode_shift<1, 3, 25,  2> { static const int value =  65; };
template <> struct vnode_shift<1, 3, 25,  3> { static const int value =  81; };
template <> struct vnode_shift<1, 3, 25,  4> { static const int value =  0; };
// ILS = 3, CHECK_NODE = 26
template <> struct vnode_shift<1, 3, 26,  0> { static const int value =  107; };
template <> struct vnode_shift<1, 3, 26,  1> { static const int value =  176; };
template <> struct vnode_shift<1, 3, 26,  2> { static const int value =  212; };
template <> struct vnode_shift<1, 3, 26,  3> { static const int value =  127; };
template <> struct vnode_shift<1, 3, 26,  4> { static const int value =  0; };
// ILS = 3, CHECK_NODE = 27
template <> struct vnode_shift<1, 3, 27,  0> { static const int value =  208; };
template <> struct vnode_shift<1, 3, 27,  1> { static const int value =  141; };
template <> struct vnode_shift<1, 3, 27,  2> { static const int value =  174; };
template <> struct vnode_shift<1, 3, 27,  3> { static const int value =  0; };
// ILS = 3, CHECK_NODE = 28
template <> struct vnode_shift<1, 3, 28,  0> { static const int value =  146; };
template <> struct vnode_shift<1, 3, 28,  1> { static const int value =  153; };
template <> struct vnode_shift<1, 3, 28,  2> { static const int value =  217; };
template <> struct vnode_shift<1, 3, 28,  3> { static const int value =  114; };
template <> struct vnode_shift<1, 3, 28,  4> { static const int value =  0; };
// ILS = 3, CHECK_NODE = 29
template <> struct vnode_shift<1, 3, 29,  0> { static const int value =  150; };
template <> struct vnode_shift<1, 3, 29,  1> { static const int value =  11; };
template <> struct vnode_shift<1, 3, 29,  2> { static const int value =  53; };
template <> struct vnode_shift<1, 3, 29,  3> { static const int value =  68; };
template <> struct vnode_shift<1, 3, 29,  4> { static const int value =  0; };
// ILS = 3, CHECK_NODE = 30
template <> struct vnode_shift<1, 3, 30,  0> { static const int value =  34; };
template <> struct vnode_shift<1, 3, 30,  1> { static const int value =  130; };
template <> struct vnode_shift<1, 3, 30,  2> { static const int value =  210; };
template <> struct vnode_shift<1, 3, 30,  3> { static const int value =  123; };
template <> struct vnode_shift<1, 3, 30,  4> { static const int value =  0; };
// ILS = 3, CHECK_NODE = 31
template <> struct vnode_shift<1, 3, 31,  0> { static const int value =  175; };
template <> struct vnode_shift<1, 3, 31,  1> { static const int value =  49; };
template <> struct vnode_shift<1, 3, 31,  2> { static const int value =  177; };
template <> struct vnode_shift<1, 3, 31,  3> { static const int value =  128; };
template <> struct vnode_shift<1, 3, 31,  4> { static const int value =  0; };
// ILS = 3, CHECK_NODE = 32
template <> struct vnode_shift<1, 3, 32,  0> { static const int value =  192; };
template <> struct vnode_shift<1, 3, 32,  1> { static const int value =  209; };
template <> struct vnode_shift<1, 3, 32,  2> { static const int value =  58; };
template <> struct vnode_shift<1, 3, 32,  3> { static const int value =  30; };
template <> struct vnode_shift<1, 3, 32,  4> { static const int value =  0; };
// ILS = 3, CHECK_NODE = 33
template <> struct vnode_shift<1, 3, 33,  0> { static const int value =  114; };
template <> struct vnode_shift<1, 3, 33,  1> { static const int value =  49; };
template <> struct vnode_shift<1, 3, 33,  2> { static const int value =  161; };
template <> struct vnode_shift<1, 3, 33,  3> { static const int value =  137; };
template <> struct vnode_shift<1, 3, 33,  4> { static const int value =  0; };
// ILS = 3, CHECK_NODE = 34
template <> struct vnode_shift<1, 3, 34,  0> { static const int value =  82; };
template <> struct vnode_shift<1, 3, 34,  1> { static const int value =  186; };
template <> struct vnode_shift<1, 3, 34,  2> { static const int value =  68; };
template <> struct vnode_shift<1, 3, 34,  3> { static const int value =  150; };
template <> struct vnode_shift<1, 3, 34,  4> { static const int value =  0; };
// ILS = 3, CHECK_NODE = 35
template <> struct vnode_shift<1, 3, 35,  0> { static const int value =  192; };
template <> struct vnode_shift<1, 3, 35,  1> { static const int value =  173; };
template <> struct vnode_shift<1, 3, 35,  2> { static const int value =  26; };
template <> struct vnode_shift<1, 3, 35,  3> { static const int value =  187; };
template <> struct vnode_shift<1, 3, 35,  4> { static const int value =  0; };
// ILS = 3, CHECK_NODE = 36
template <> struct vnode_shift<1, 3, 36,  0> { static const int value =  222; };
template <> struct vnode_shift<1, 3, 36,  1> { static const int value =  157; };
template <> struct vnode_shift<1, 3, 36,  2> { static const int value =  0; };
template <> struct vnode_shift<1, 3, 36,  3> { static const int value =  6; };
template <> struct vnode_shift<1, 3, 36,  4> { static const int value =  0; };
// ILS = 3, CHECK_NODE = 37
template <> struct vnode_shift<1, 3, 37,  0> { static const int value =  81; };
template <> struct vnode_shift<1, 3, 37,  1> { static const int value =  195; };
template <> struct vnode_shift<1, 3, 37,  2> { static const int value =  138; };
template <> struct vnode_shift<1, 3, 37,  3> { static const int value =  0; };
// ILS = 3, CHECK_NODE = 38
template <> struct vnode_shift<1, 3, 38,  0> { static const int value =  123; };
template <> struct vnode_shift<1, 3, 38,  1> { static const int value =  90; };
template <> struct vnode_shift<1, 3, 38,  2> { static const int value =  73; };
template <> struct vnode_shift<1, 3, 38,  3> { static const int value =  10; };
template <> struct vnode_shift<1, 3, 38,  4> { static const int value =  0; };
// ILS = 3, CHECK_NODE = 39
template <> struct vnode_shift<1, 3, 39,  0> { static const int value =  12; };
template <> struct vnode_shift<1, 3, 39,  1> { static const int value =  77; };
template <> struct vnode_shift<1, 3, 39,  2> { static const int value =  49; };
template <> struct vnode_shift<1, 3, 39,  3> { static const int value =  114; };
template <> struct vnode_shift<1, 3, 39,  4> { static const int value =  0; };
// ILS = 3, CHECK_NODE = 40
template <> struct vnode_shift<1, 3, 40,  0> { static const int value =  67; };
template <> struct vnode_shift<1, 3, 40,  1> { static const int value =  45; };
template <> struct vnode_shift<1, 3, 40,  2> { static const int value =  96; };
template <> struct vnode_shift<1, 3, 40,  3> { static const int value =  0; };
// ILS = 3, CHECK_NODE = 41
template <> struct vnode_shift<1, 3, 41,  0> { static const int value =  23; };
template <> struct vnode_shift<1, 3, 41,  1> { static const int value =  215; };
template <> struct vnode_shift<1, 3, 41,  2> { static const int value =  60; };
template <> struct vnode_shift<1, 3, 41,  3> { static const int value =  167; };
template <> struct vnode_shift<1, 3, 41,  4> { static const int value =  0; };
// ILS = 3, CHECK_NODE = 42
template <> struct vnode_shift<1, 3, 42,  0> { static const int value =  114; };
template <> struct vnode_shift<1, 3, 42,  1> { static const int value =  91; };
template <> struct vnode_shift<1, 3, 42,  2> { static const int value =  78; };
template <> struct vnode_shift<1, 3, 42,  3> { static const int value =  0; };
// ILS = 3, CHECK_NODE = 43
template <> struct vnode_shift<1, 3, 43,  0> { static const int value =  206; };
template <> struct vnode_shift<1, 3, 43,  1> { static const int value =  22; };
template <> struct vnode_shift<1, 3, 43,  2> { static const int value =  134; };
template <> struct vnode_shift<1, 3, 43,  3> { static const int value =  161; };
template <> struct vnode_shift<1, 3, 43,  4> { static const int value =  0; };
// ILS = 3, CHECK_NODE = 44
template <> struct vnode_shift<1, 3, 44,  0> { static const int value =  84; };
template <> struct vnode_shift<1, 3, 44,  1> { static const int value =  4; };
template <> struct vnode_shift<1, 3, 44,  2> { static const int value =  9; };
template <> struct vnode_shift<1, 3, 44,  3> { static const int value =  12; };
template <> struct vnode_shift<1, 3, 44,  4> { static const int value =  0; };
// ILS = 3, CHECK_NODE = 45
template <> struct vnode_shift<1, 3, 45,  0> { static const int value =  184; };
template <> struct vnode_shift<1, 3, 45,  1> { static const int value =  121; };
template <> struct vnode_shift<1, 3, 45,  2> { static const int value =  29; };
template <> struct vnode_shift<1, 3, 45,  3> { static const int value =  0; };
// ILS = 4, CHECK_NODE = 0
template <> struct vnode_shift<1, 4, 0,  0> { static const int value =  211; };
template <> struct vnode_shift<1, 4, 0,  1> { static const int value =  198; };
template <> struct vnode_shift<1, 4, 0,  2> { static const int value =  188; };
template <> struct vnode_shift<1, 4, 0,  3> { static const int value =  186; };
template <> struct vnode_shift<1, 4, 0,  4> { static const int value =  219; };
template <> struct vnode_shift<1, 4, 0,  5> { static const int value =  4; };
template <> struct vnode_shift<1, 4, 0,  6> { static const int value =  29; };
template <> struct vnode_shift<1, 4, 0,  7> { static const int value =  144; };
template <> struct vnode_shift<1, 4, 0,  8> { static const int value =  116; };
template <> struct vnode_shift<1, 4, 0,  9> { static const int value =  216; };
template <> struct vnode_shift<1, 4, 0,  10> { static const int value =  115; };
template <> struct vnode_shift<1, 4, 0,  11> { static const int value =  233; };
template <> struct vnode_shift<1, 4, 0,  12> { static const int value =  144; };
template <> struct vnode_shift<1, 4, 0,  13> { static const int value =  95; };
template <> struct vnode_shift<1, 4, 0,  14> { static const int value =  216; };
template <> struct vnode_shift<1, 4, 0,  15> { static const int value =  73; };
template <> struct vnode_shift<1, 4, 0,  16> { static const int value =  261; };
template <> struct vnode_shift<1, 4, 0,  17> { static const int value =  1; };
template <> struct vnode_shift<1, 4, 0,  18> { static const int value =  0; };
// ILS = 4, CHECK_NODE = 1
template <> struct vnode_shift<1, 4, 1,  0> { static const int value =  179; };
template <> struct vnode_shift<1, 4, 1,  1> { static const int value =  162; };
template <> struct vnode_shift<1, 4, 1,  2> { static const int value =  223; };
template <> struct vnode_shift<1, 4, 1,  3> { static const int value =  256; };
template <> struct vnode_shift<1, 4, 1,  4> { static const int value =  160; };
template <> struct vnode_shift<1, 4, 1,  5> { static const int value =  76; };
template <> struct vnode_shift<1, 4, 1,  6> { static const int value =  202; };
template <> struct vnode_shift<1, 4, 1,  7> { static const int value =  117; };
template <> struct vnode_shift<1, 4, 1,  8> { static const int value =  109; };
template <> struct vnode_shift<1, 4, 1,  9> { static const int value =  15; };
template <> struct vnode_shift<1, 4, 1,  10> { static const int value =  72; };
template <> struct vnode_shift<1, 4, 1,  11> { static const int value =  152; };
template <> struct vnode_shift<1, 4, 1,  12> { static const int value =  158; };
template <> struct vnode_shift<1, 4, 1,  13> { static const int value =  147; };
template <> struct vnode_shift<1, 4, 1,  14> { static const int value =  156; };
template <> struct vnode_shift<1, 4, 1,  15> { static const int value =  119; };
template <> struct vnode_shift<1, 4, 1,  16> { static const int value =  0; };
template <> struct vnode_shift<1, 4, 1,  17> { static const int value =  0; };
template <> struct vnode_shift<1, 4, 1,  18> { static const int value =  0; };
// ILS = 4, CHECK_NODE = 2
template <> struct vnode_shift<1, 4, 2,  0> { static const int value =  258; };
template <> struct vnode_shift<1, 4, 2,  1> { static const int value =  167; };
template <> struct vnode_shift<1, 4, 2,  2> { static const int value =  220; };
template <> struct vnode_shift<1, 4, 2,  3> { static const int value =  133; };
template <> struct vnode_shift<1, 4, 2,  4> { static const int value =  243; };
template <> struct vnode_shift<1, 4, 2,  5> { static const int value =  202; };
template <> struct vnode_shift<1, 4, 2,  6> { static const int value =  218; };
template <> struct vnode_shift<1, 4, 2,  7> { static const int value =  63; };
template <> struct vnode_shift<1, 4, 2,  8> { static const int value =  0; };
template <> struct vnode_shift<1, 4, 2,  9> { static const int value =  3; };
template <> struct vnode_shift<1, 4, 2,  10> { static const int value =  74; };
template <> struct vnode_shift<1, 4, 2,  11> { static const int value =  229; };
template <> struct vnode_shift<1, 4, 2,  12> { static const int value =  0; };
template <> struct vnode_shift<1, 4, 2,  13> { static const int value =  216; };
template <> struct vnode_shift<1, 4, 2,  14> { static const int value =  269; };
template <> struct vnode_shift<1, 4, 2,  15> { static const int value =  200; };
template <> struct vnode_shift<1, 4, 2,  16> { static const int value =  234; };
template <> struct vnode_shift<1, 4, 2,  17> { static const int value =  0; };
template <> struct vnode_shift<1, 4, 2,  18> { static const int value =  0; };
// ILS = 4, CHECK_NODE = 3
template <> struct vnode_shift<1, 4, 3,  0> { static const int value =  187; };
template <> struct vnode_shift<1, 4, 3,  1> { static const int value =  145; };
template <> struct vnode_shift<1, 4, 3,  2> { static const int value =  166; };
template <> struct vnode_shift<1, 4, 3,  3> { static const int value =  108; };
template <> struct vnode_shift<1, 4, 3,  4> { static const int value =  82; };
template <> struct vnode_shift<1, 4, 3,  5> { static const int value =  132; };
template <> struct vnode_shift<1, 4, 3,  6> { static const int value =  197; };
template <> struct vnode_shift<1, 4, 3,  7> { static const int value =  41; };
template <> struct vnode_shift<1, 4, 3,  8> { static const int value =  162; };
template <> struct vnode_shift<1, 4, 3,  9> { static const int value =  57; };
template <> struct vnode_shift<1, 4, 3,  10> { static const int value =  36; };
template <> struct vnode_shift<1, 4, 3,  11> { static const int value =  115; };
template <> struct vnode_shift<1, 4, 3,  12> { static const int value =  242; };
template <> struct vnode_shift<1, 4, 3,  13> { static const int value =  165; };
template <> struct vnode_shift<1, 4, 3,  14> { static const int value =  0; };
template <> struct vnode_shift<1, 4, 3,  15> { static const int value =  113; };
template <> struct vnode_shift<1, 4, 3,  16> { static const int value =  108; };
template <> struct vnode_shift<1, 4, 3,  17> { static const int value =  1; };
template <> struct vnode_shift<1, 4, 3,  18> { static const int value =  0; };
// ILS = 4, CHECK_NODE = 4
template <> struct vnode_shift<1, 4, 4,  0> { static const int value =  246; };
template <> struct vnode_shift<1, 4, 4,  1> { static const int value =  235; };
template <> struct vnode_shift<1, 4, 4,  2> { static const int value =  0; };
// ILS = 4, CHECK_NODE = 5
template <> struct vnode_shift<1, 4, 5,  0> { static const int value =  261; };
template <> struct vnode_shift<1, 4, 5,  1> { static const int value =  181; };
template <> struct vnode_shift<1, 4, 5,  2> { static const int value =  72; };
template <> struct vnode_shift<1, 4, 5,  3> { static const int value =  283; };
template <> struct vnode_shift<1, 4, 5,  4> { static const int value =  254; };
template <> struct vnode_shift<1, 4, 5,  5> { static const int value =  79; };
template <> struct vnode_shift<1, 4, 5,  6> { static const int value =  144; };
template <> struct vnode_shift<1, 4, 5,  7> { static const int value =  0; };
// ILS = 4, CHECK_NODE = 6
template <> struct vnode_shift<1, 4, 6,  0> { static const int value =  80; };
template <> struct vnode_shift<1, 4, 6,  1> { static const int value =  144; };
template <> struct vnode_shift<1, 4, 6,  2> { static const int value =  169; };
template <> struct vnode_shift<1, 4, 6,  3> { static const int value =  90; };
template <> struct vnode_shift<1, 4, 6,  4> { static const int value =  59; };
template <> struct vnode_shift<1, 4, 6,  5> { static const int value =  177; };
template <> struct vnode_shift<1, 4, 6,  6> { static const int value =  151; };
template <> struct vnode_shift<1, 4, 6,  7> { static const int value =  108; };
template <> struct vnode_shift<1, 4, 6,  8> { static const int value =  0; };
// ILS = 4, CHECK_NODE = 7
template <> struct vnode_shift<1, 4, 7,  0> { static const int value =  169; };
template <> struct vnode_shift<1, 4, 7,  1> { static const int value =  189; };
template <> struct vnode_shift<1, 4, 7,  2> { static const int value =  154; };
template <> struct vnode_shift<1, 4, 7,  3> { static const int value =  184; };
template <> struct vnode_shift<1, 4, 7,  4> { static const int value =  104; };
template <> struct vnode_shift<1, 4, 7,  5> { static const int value =  164; };
template <> struct vnode_shift<1, 4, 7,  6> { static const int value =  0; };
// ILS = 4, CHECK_NODE = 8
template <> struct vnode_shift<1, 4, 8,  0> { static const int value =  54; };
template <> struct vnode_shift<1, 4, 8,  1> { static const int value =  0; };
template <> struct vnode_shift<1, 4, 8,  2> { static const int value =  252; };
template <> struct vnode_shift<1, 4, 8,  3> { static const int value =  41; };
template <> struct vnode_shift<1, 4, 8,  4> { static const int value =  98; };
template <> struct vnode_shift<1, 4, 8,  5> { static const int value =  46; };
template <> struct vnode_shift<1, 4, 8,  6> { static const int value =  15; };
template <> struct vnode_shift<1, 4, 8,  7> { static const int value =  230; };
template <> struct vnode_shift<1, 4, 8,  8> { static const int value =  54; };
template <> struct vnode_shift<1, 4, 8,  9> { static const int value =  0; };
// ILS = 4, CHECK_NODE = 9
template <> struct vnode_shift<1, 4, 9,  0> { static const int value =  162; };
template <> struct vnode_shift<1, 4, 9,  1> { static const int value =  159; };
template <> struct vnode_shift<1, 4, 9,  2> { static const int value =  93; };
template <> struct vnode_shift<1, 4, 9,  3> { static const int value =  134; };
template <> struct vnode_shift<1, 4, 9,  4> { static const int value =  45; };
template <> struct vnode_shift<1, 4, 9,  5> { static const int value =  132; };
template <> struct vnode_shift<1, 4, 9,  6> { static const int value =  76; };
template <> struct vnode_shift<1, 4, 9,  7> { static const int value =  209; };
template <> struct vnode_shift<1, 4, 9,  8> { static const int value =  0; };
// ILS = 4, CHECK_NODE = 10
template <> struct vnode_shift<1, 4, 10,  0> { static const int value =  178; };
template <> struct vnode_shift<1, 4, 10,  1> { static const int value =  1; };
template <> struct vnode_shift<1, 4, 10,  2> { static const int value =  28; };
template <> struct vnode_shift<1, 4, 10,  3> { static const int value =  267; };
template <> struct vnode_shift<1, 4, 10,  4> { static const int value =  234; };
template <> struct vnode_shift<1, 4, 10,  5> { static const int value =  201; };
template <> struct vnode_shift<1, 4, 10,  6> { static const int value =  0; };
// ILS = 4, CHECK_NODE = 11
template <> struct vnode_shift<1, 4, 11,  0> { static const int value =  55; };
template <> struct vnode_shift<1, 4, 11,  1> { static const int value =  23; };
template <> struct vnode_shift<1, 4, 11,  2> { static const int value =  274; };
template <> struct vnode_shift<1, 4, 11,  3> { static const int value =  181; };
template <> struct vnode_shift<1, 4, 11,  4> { static const int value =  273; };
template <> struct vnode_shift<1, 4, 11,  5> { static const int value =  39; };
template <> struct vnode_shift<1, 4, 11,  6> { static const int value =  26; };
template <> struct vnode_shift<1, 4, 11,  7> { static const int value =  0; };
// ILS = 4, CHECK_NODE = 12
template <> struct vnode_shift<1, 4, 12,  0> { static const int value =  225; };
template <> struct vnode_shift<1, 4, 12,  1> { static const int value =  162; };
template <> struct vnode_shift<1, 4, 12,  2> { static const int value =  244; };
template <> struct vnode_shift<1, 4, 12,  3> { static const int value =  151; };
template <> struct vnode_shift<1, 4, 12,  4> { static const int value =  238; };
template <> struct vnode_shift<1, 4, 12,  5> { static const int value =  243; };
template <> struct vnode_shift<1, 4, 12,  6> { static const int value =  0; };
// ILS = 4, CHECK_NODE = 13
template <> struct vnode_shift<1, 4, 13,  0> { static const int value =  231; };
template <> struct vnode_shift<1, 4, 13,  1> { static const int value =  0; };
template <> struct vnode_shift<1, 4, 13,  2> { static const int value =  216; };
template <> struct vnode_shift<1, 4, 13,  3> { static const int value =  47; };
template <> struct vnode_shift<1, 4, 13,  4> { static const int value =  36; };
template <> struct vnode_shift<1, 4, 13,  5> { static const int value =  0; };
// ILS = 4, CHECK_NODE = 14
template <> struct vnode_shift<1, 4, 14,  0> { static const int value =  0; };
template <> struct vnode_shift<1, 4, 14,  1> { static const int value =  186; };
template <> struct vnode_shift<1, 4, 14,  2> { static const int value =  253; };
template <> struct vnode_shift<1, 4, 14,  3> { static const int value =  16; };
template <> struct vnode_shift<1, 4, 14,  4> { static const int value =  0; };
template <> struct vnode_shift<1, 4, 14,  5> { static const int value =  79; };
template <> struct vnode_shift<1, 4, 14,  6> { static const int value =  0; };
// ILS = 4, CHECK_NODE = 15
template <> struct vnode_shift<1, 4, 15,  0> { static const int value =  170; };
template <> struct vnode_shift<1, 4, 15,  1> { static const int value =  0; };
template <> struct vnode_shift<1, 4, 15,  2> { static const int value =  183; };
template <> struct vnode_shift<1, 4, 15,  3> { static const int value =  108; };
template <> struct vnode_shift<1, 4, 15,  4> { static const int value =  68; };
template <> struct vnode_shift<1, 4, 15,  5> { static const int value =  64; };
template <> struct vnode_shift<1, 4, 15,  6> { static const int value =  0; };
// ILS = 4, CHECK_NODE = 16
template <> struct vnode_shift<1, 4, 16,  0> { static const int value =  270; };
template <> struct vnode_shift<1, 4, 16,  1> { static const int value =  13; };
template <> struct vnode_shift<1, 4, 16,  2> { static const int value =  99; };
template <> struct vnode_shift<1, 4, 16,  3> { static const int value =  54; };
template <> struct vnode_shift<1, 4, 16,  4> { static const int value =  0; };
template <> struct vnode_shift<1, 4, 16,  5> { static const int value =  0; };
// ILS = 4, CHECK_NODE = 17
template <> struct vnode_shift<1, 4, 17,  0> { static const int value =  153; };
template <> struct vnode_shift<1, 4, 17,  1> { static const int value =  137; };
template <> struct vnode_shift<1, 4, 17,  2> { static const int value =  0; };
template <> struct vnode_shift<1, 4, 17,  3> { static const int value =  0; };
template <> struct vnode_shift<1, 4, 17,  4> { static const int value =  162; };
template <> struct vnode_shift<1, 4, 17,  5> { static const int value =  0; };
// ILS = 4, CHECK_NODE = 18
template <> struct vnode_shift<1, 4, 18,  0> { static const int value =  161; };
template <> struct vnode_shift<1, 4, 18,  1> { static const int value =  151; };
template <> struct vnode_shift<1, 4, 18,  2> { static const int value =  0; };
template <> struct vnode_shift<1, 4, 18,  3> { static const int value =  241; };
template <> struct vnode_shift<1, 4, 18,  4> { static const int value =  144; };
template <> struct vnode_shift<1, 4, 18,  5> { static const int value =  0; };
// ILS = 4, CHECK_NODE = 19
template <> struct vnode_shift<1, 4, 19,  0> { static const int value =  0; };
template <> struct vnode_shift<1, 4, 19,  1> { static const int value =  0; };
template <> struct vnode_shift<1, 4, 19,  2> { static const int value =  118; };
template <> struct vnode_shift<1, 4, 19,  3> { static const int value =  144; };
template <> struct vnode_shift<1, 4, 19,  4> { static const int value =  0; };
template <> struct vnode_shift<1, 4, 19,  5> { static const int value =  0; };
// ILS = 4, CHECK_NODE = 20
template <> struct vnode_shift<1, 4, 20,  0> { static const int value =  265; };
template <> struct vnode_shift<1, 4, 20,  1> { static const int value =  81; };
template <> struct vnode_shift<1, 4, 20,  2> { static const int value =  90; };
template <> struct vnode_shift<1, 4, 20,  3> { static const int value =  144; };
template <> struct vnode_shift<1, 4, 20,  4> { static const int value =  228; };
template <> struct vnode_shift<1, 4, 20,  5> { static const int value =  0; };
// ILS = 4, CHECK_NODE = 21
template <> struct vnode_shift<1, 4, 21,  0> { static const int value =  64; };
template <> struct vnode_shift<1, 4, 21,  1> { static const int value =  46; };
template <> struct vnode_shift<1, 4, 21,  2> { static const int value =  266; };
template <> struct vnode_shift<1, 4, 21,  3> { static const int value =  9; };
template <> struct vnode_shift<1, 4, 21,  4> { static const int value =  18; };
template <> struct vnode_shift<1, 4, 21,  5> { static const int value =  0; };
// ILS = 4, CHECK_NODE = 22
template <> struct vnode_shift<1, 4, 22,  0> { static const int value =  72; };
template <> struct vnode_shift<1, 4, 22,  1> { static const int value =  189; };
template <> struct vnode_shift<1, 4, 22,  2> { static const int value =  72; };
template <> struct vnode_shift<1, 4, 22,  3> { static const int value =  257; };
template <> struct vnode_shift<1, 4, 22,  4> { static const int value =  0; };
// ILS = 4, CHECK_NODE = 23
template <> struct vnode_shift<1, 4, 23,  0> { static const int value =  180; };
template <> struct vnode_shift<1, 4, 23,  1> { static const int value =  0; };
template <> struct vnode_shift<1, 4, 23,  2> { static const int value =  0; };
template <> struct vnode_shift<1, 4, 23,  3> { static const int value =  165; };
template <> struct vnode_shift<1, 4, 23,  4> { static const int value =  0; };
// ILS = 4, CHECK_NODE = 24
template <> struct vnode_shift<1, 4, 24,  0> { static const int value =  236; };
template <> struct vnode_shift<1, 4, 24,  1> { static const int value =  199; };
template <> struct vnode_shift<1, 4, 24,  2> { static const int value =  0; };
template <> struct vnode_shift<1, 4, 24,  3> { static const int value =  266; };
template <> struct vnode_shift<1, 4, 24,  4> { static const int value =  0; };
template <> struct vnode_shift<1, 4, 24,  5> { static const int value =  0; };
// ILS = 4, CHECK_NODE = 25
template <> struct vnode_shift<1, 4, 25,  0> { static const int value =  205; };
template <> struct vnode_shift<1, 4, 25,  1> { static const int value =  0; };
template <> struct vnode_shift<1, 4, 25,  2> { static const int value =  0; };
template <> struct vnode_shift<1, 4, 25,  3> { static const int value =  183; };
template <> struct vnode_shift<1, 4, 25,  4> { static const int value =  0; };
// ILS = 4, CHECK_NODE = 26
template <> struct vnode_shift<1, 4, 26,  0> { static const int value =  0; };
template <> struct vnode_shift<1, 4, 26,  1> { static const int value =  0; };
template <> struct vnode_shift<1, 4, 26,  2> { static const int value =  0; };
template <> struct vnode_shift<1, 4, 26,  3> { static const int value =  277; };
template <> struct vnode_shift<1, 4, 26,  4> { static const int value =  0; };
// ILS = 4, CHECK_NODE = 27
template <> struct vnode_shift<1, 4, 27,  0> { static const int value =  45; };
template <> struct vnode_shift<1, 4, 27,  1> { static const int value =  36; };
template <> struct vnode_shift<1, 4, 27,  2> { static const int value =  72; };
template <> struct vnode_shift<1, 4, 27,  3> { static const int value =  0; };
// ILS = 4, CHECK_NODE = 28
template <> struct vnode_shift<1, 4, 28,  0> { static const int value =  275; };
template <> struct vnode_shift<1, 4, 28,  1> { static const int value =  0; };
template <> struct vnode_shift<1, 4, 28,  2> { static const int value =  155; };
template <> struct vnode_shift<1, 4, 28,  3> { static const int value =  62; };
template <> struct vnode_shift<1, 4, 28,  4> { static const int value =  0; };
// ILS = 4, CHECK_NODE = 29
template <> struct vnode_shift<1, 4, 29,  0> { static const int value =  0; };
template <> struct vnode_shift<1, 4, 29,  1> { static const int value =  180; };
template <> struct vnode_shift<1, 4, 29,  2> { static const int value =  0; };
template <> struct vnode_shift<1, 4, 29,  3> { static const int value =  42; };
template <> struct vnode_shift<1, 4, 29,  4> { static const int value =  0; };
// ILS = 4, CHECK_NODE = 30
template <> struct vnode_shift<1, 4, 30,  0> { static const int value =  0; };
template <> struct vnode_shift<1, 4, 30,  1> { static const int value =  90; };
template <> struct vnode_shift<1, 4, 30,  2> { static const int value =  252; };
template <> struct vnode_shift<1, 4, 30,  3> { static const int value =  173; };
template <> struct vnode_shift<1, 4, 30,  4> { static const int value =  0; };
// ILS = 4, CHECK_NODE = 31
template <> struct vnode_shift<1, 4, 31,  0> { static const int value =  144; };
template <> struct vnode_shift<1, 4, 31,  1> { static const int value =  144; };
template <> struct vnode_shift<1, 4, 31,  2> { static const int value =  166; };
template <> struct vnode_shift<1, 4, 31,  3> { static const int value =  19; };
template <> struct vnode_shift<1, 4, 31,  4> { static const int value =  0; };
// ILS = 4, CHECK_NODE = 32
template <> struct vnode_shift<1, 4, 32,  0> { static const int value =  0; };
template <> struct vnode_shift<1, 4, 32,  1> { static const int value =  211; };
template <> struct vnode_shift<1, 4, 32,  2> { static const int value =  36; };
template <> struct vnode_shift<1, 4, 32,  3> { static const int value =  162; };
template <> struct vnode_shift<1, 4, 32,  4> { static const int value =  0; };
// ILS = 4, CHECK_NODE = 33
template <> struct vnode_shift<1, 4, 33,  0> { static const int value =  0; };
template <> struct vnode_shift<1, 4, 33,  1> { static const int value =  0; };
template <> struct vnode_shift<1, 4, 33,  2> { static const int value =  76; };
template <> struct vnode_shift<1, 4, 33,  3> { static const int value =  18; };
template <> struct vnode_shift<1, 4, 33,  4> { static const int value =  0; };
// ILS = 4, CHECK_NODE = 34
template <> struct vnode_shift<1, 4, 34,  0> { static const int value =  197; };
template <> struct vnode_shift<1, 4, 34,  1> { static const int value =  0; };
template <> struct vnode_shift<1, 4, 34,  2> { static const int value =  108; };
template <> struct vnode_shift<1, 4, 34,  3> { static const int value =  0; };
template <> struct vnode_shift<1, 4, 34,  4> { static const int value =  0; };
// ILS = 4, CHECK_NODE = 35
template <> struct vnode_shift<1, 4, 35,  0> { static const int value =  199; };
template <> struct vnode_shift<1, 4, 35,  1> { static const int value =  278; };
template <> struct vnode_shift<1, 4, 35,  2> { static const int value =  0; };
template <> struct vnode_shift<1, 4, 35,  3> { static const int value =  205; };
template <> struct vnode_shift<1, 4, 35,  4> { static const int value =  0; };
// ILS = 4, CHECK_NODE = 36
template <> struct vnode_shift<1, 4, 36,  0> { static const int value =  216; };
template <> struct vnode_shift<1, 4, 36,  1> { static const int value =  16; };
template <> struct vnode_shift<1, 4, 36,  2> { static const int value =  0; };
template <> struct vnode_shift<1, 4, 36,  3> { static const int value =  0; };
template <> struct vnode_shift<1, 4, 36,  4> { static const int value =  0; };
// ILS = 4, CHECK_NODE = 37
template <> struct vnode_shift<1, 4, 37,  0> { static const int value =  72; };
template <> struct vnode_shift<1, 4, 37,  1> { static const int value =  144; };
template <> struct vnode_shift<1, 4, 37,  2> { static const int value =  0; };
template <> struct vnode_shift<1, 4, 37,  3> { static const int value =  0; };
// ILS = 4, CHECK_NODE = 38
template <> struct vnode_shift<1, 4, 38,  0> { static const int value =  190; };
template <> struct vnode_shift<1, 4, 38,  1> { static const int value =  0; };
template <> struct vnode_shift<1, 4, 38,  2> { static const int value =  0; };
template <> struct vnode_shift<1, 4, 38,  3> { static const int value =  0; };
template <> struct vnode_shift<1, 4, 38,  4> { static const int value =  0; };
// ILS = 4, CHECK_NODE = 39
template <> struct vnode_shift<1, 4, 39,  0> { static const int value =  153; };
template <> struct vnode_shift<1, 4, 39,  1> { static const int value =  0; };
template <> struct vnode_shift<1, 4, 39,  2> { static const int value =  165; };
template <> struct vnode_shift<1, 4, 39,  3> { static const int value =  117; };
template <> struct vnode_shift<1, 4, 39,  4> { static const int value =  0; };
// ILS = 4, CHECK_NODE = 40
template <> struct vnode_shift<1, 4, 40,  0> { static const int value =  216; };
template <> struct vnode_shift<1, 4, 40,  1> { static const int value =  144; };
template <> struct vnode_shift<1, 4, 40,  2> { static const int value =  2; };
template <> struct vnode_shift<1, 4, 40,  3> { static const int value =  0; };
// ILS = 4, CHECK_NODE = 41
template <> struct vnode_shift<1, 4, 41,  0> { static const int value =  0; };
template <> struct vnode_shift<1, 4, 41,  1> { static const int value =  0; };
template <> struct vnode_shift<1, 4, 41,  2> { static const int value =  0; };
template <> struct vnode_shift<1, 4, 41,  3> { static const int value =  183; };
template <> struct vnode_shift<1, 4, 41,  4> { static const int value =  0; };
// ILS = 4, CHECK_NODE = 42
template <> struct vnode_shift<1, 4, 42,  0> { static const int value =  27; };
template <> struct vnode_shift<1, 4, 42,  1> { static const int value =  0; };
template <> struct vnode_shift<1, 4, 42,  2> { static const int value =  35; };
template <> struct vnode_shift<1, 4, 42,  3> { static const int value =  0; };
// ILS = 4, CHECK_NODE = 43
template <> struct vnode_shift<1, 4, 43,  0> { static const int value =  52; };
template <> struct vnode_shift<1, 4, 43,  1> { static const int value =  243; };
template <> struct vnode_shift<1, 4, 43,  2> { static const int value =  0; };
template <> struct vnode_shift<1, 4, 43,  3> { static const int value =  270; };
template <> struct vnode_shift<1, 4, 43,  4> { static const int value =  0; };
// ILS = 4, CHECK_NODE = 44
template <> struct vnode_shift<1, 4, 44,  0> { static const int value =  18; };
template <> struct vnode_shift<1, 4, 44,  1> { static const int value =  0; };
template <> struct vnode_shift<1, 4, 44,  2> { static const int value =  0; };
template <> struct vnode_shift<1, 4, 44,  3> { static const int value =  57; };
template <> struct vnode_shift<1, 4, 44,  4> { static const int value =  0; };
// ILS = 4, CHECK_NODE = 45
template <> struct vnode_shift<1, 4, 45,  0> { static const int value =  168; };
template <> struct vnode_shift<1, 4, 45,  1> { static const int value =  0; };
template <> struct vnode_shift<1, 4, 45,  2> { static const int value =  144; };
template <> struct vnode_shift<1, 4, 45,  3> { static const int value =  0; };
// ILS = 5, CHECK_NODE = 0
template <> struct vnode_shift<1, 5, 0,  0> { static const int value =  294; };
template <> struct vnode_shift<1, 5, 0,  1> { static const int value =  118; };
template <> struct vnode_shift<1, 5, 0,  2> { static const int value =  167; };
template <> struct vnode_shift<1, 5, 0,  3> { static const int value =  330; };
template <> struct vnode_shift<1, 5, 0,  4> { static const int value =  207; };
template <> struct vnode_shift<1, 5, 0,  5> { static const int value =  165; };
template <> struct vnode_shift<1, 5, 0,  6> { static const int value =  243; };
template <> struct vnode_shift<1, 5, 0,  7> { static const int value =  250; };
template <> struct vnode_shift<1, 5, 0,  8> { static const int value =  1; };
template <> struct vnode_shift<1, 5, 0,  9> { static const int value =  339; };
template <> struct vnode_shift<1, 5, 0,  10> { static const int value =  201; };
template <> struct vnode_shift<1, 5, 0,  11> { static const int value =  53; };
template <> struct vnode_shift<1, 5, 0,  12> { static const int value =  347; };
template <> struct vnode_shift<1, 5, 0,  13> { static const int value =  304; };
template <> struct vnode_shift<1, 5, 0,  14> { static const int value =  167; };
template <> struct vnode_shift<1, 5, 0,  15> { static const int value =  47; };
template <> struct vnode_shift<1, 5, 0,  16> { static const int value =  188; };
template <> struct vnode_shift<1, 5, 0,  17> { static const int value =  1; };
template <> struct vnode_shift<1, 5, 0,  18> { static const int value =  0; };
// ILS = 5, CHECK_NODE = 1
template <> struct vnode_shift<1, 5, 1,  0> { static const int value =  77; };
template <> struct vnode_shift<1, 5, 1,  1> { static const int value =  225; };
template <> struct vnode_shift<1, 5, 1,  2> { static const int value =  96; };
template <> struct vnode_shift<1, 5, 1,  3> { static const int value =  338; };
template <> struct vnode_shift<1, 5, 1,  4> { static const int value =  268; };
template <> struct vnode_shift<1, 5, 1,  5> { static const int value =  112; };
template <> struct vnode_shift<1, 5, 1,  6> { static const int value =  302; };
template <> struct vnode_shift<1, 5, 1,  7> { static const int value =  50; };
template <> struct vnode_shift<1, 5, 1,  8> { static const int value =  167; };
template <> struct vnode_shift<1, 5, 1,  9> { static const int value =  253; };
template <> struct vnode_shift<1, 5, 1,  10> { static const int value =  334; };
template <> struct vnode_shift<1, 5, 1,  11> { static const int value =  242; };
template <> struct vnode_shift<1, 5, 1,  12> { static const int value =  257; };
template <> struct vnode_shift<1, 5, 1,  13> { static const int value =  133; };
template <> struct vnode_shift<1, 5, 1,  14> { static const int value =  9; };
template <> struct vnode_shift<1, 5, 1,  15> { static const int value =  302; };
template <> struct vnode_shift<1, 5, 1,  16> { static const int value =  0; };
template <> struct vnode_shift<1, 5, 1,  17> { static const int value =  0; };
template <> struct vnode_shift<1, 5, 1,  18> { static const int value =  0; };
// ILS = 5, CHECK_NODE = 2
template <> struct vnode_shift<1, 5, 2,  0> { static const int value =  226; };
template <> struct vnode_shift<1, 5, 2,  1> { static const int value =  35; };
template <> struct vnode_shift<1, 5, 2,  2> { static const int value =  213; };
template <> struct vnode_shift<1, 5, 2,  3> { static const int value =  302; };
template <> struct vnode_shift<1, 5, 2,  4> { static const int value =  111; };
template <> struct vnode_shift<1, 5, 2,  5> { static const int value =  265; };
template <> struct vnode_shift<1, 5, 2,  6> { static const int value =  128; };
template <> struct vnode_shift<1, 5, 2,  7> { static const int value =  237; };
template <> struct vnode_shift<1, 5, 2,  8> { static const int value =  294; };
template <> struct vnode_shift<1, 5, 2,  9> { static const int value =  127; };
template <> struct vnode_shift<1, 5, 2,  10> { static const int value =  110; };
template <> struct vnode_shift<1, 5, 2,  11> { static const int value =  286; };
template <> struct vnode_shift<1, 5, 2,  12> { static const int value =  125; };
template <> struct vnode_shift<1, 5, 2,  13> { static const int value =  131; };
template <> struct vnode_shift<1, 5, 2,  14> { static const int value =  163; };
template <> struct vnode_shift<1, 5, 2,  15> { static const int value =  210; };
template <> struct vnode_shift<1, 5, 2,  16> { static const int value =  7; };
template <> struct vnode_shift<1, 5, 2,  17> { static const int value =  0; };
template <> struct vnode_shift<1, 5, 2,  18> { static const int value =  0; };
// ILS = 5, CHECK_NODE = 3
template <> struct vnode_shift<1, 5, 3,  0> { static const int value =  97; };
template <> struct vnode_shift<1, 5, 3,  1> { static const int value =  94; };
template <> struct vnode_shift<1, 5, 3,  2> { static const int value =  49; };
template <> struct vnode_shift<1, 5, 3,  3> { static const int value =  279; };
template <> struct vnode_shift<1, 5, 3,  4> { static const int value =  139; };
template <> struct vnode_shift<1, 5, 3,  5> { static const int value =  166; };
template <> struct vnode_shift<1, 5, 3,  6> { static const int value =  91; };
template <> struct vnode_shift<1, 5, 3,  7> { static const int value =  106; };
template <> struct vnode_shift<1, 5, 3,  8> { static const int value =  246; };
template <> struct vnode_shift<1, 5, 3,  9> { static const int value =  345; };
template <> struct vnode_shift<1, 5, 3,  10> { static const int value =  269; };
template <> struct vnode_shift<1, 5, 3,  11> { static const int value =  185; };
template <> struct vnode_shift<1, 5, 3,  12> { static const int value =  249; };
template <> struct vnode_shift<1, 5, 3,  13> { static const int value =  215; };
template <> struct vnode_shift<1, 5, 3,  14> { static const int value =  143; };
template <> struct vnode_shift<1, 5, 3,  15> { static const int value =  121; };
template <> struct vnode_shift<1, 5, 3,  16> { static const int value =  121; };
template <> struct vnode_shift<1, 5, 3,  17> { static const int value =  1; };
template <> struct vnode_shift<1, 5, 3,  18> { static const int value =  0; };
// ILS = 5, CHECK_NODE = 4
template <> struct vnode_shift<1, 5, 4,  0> { static const int value =  42; };
template <> struct vnode_shift<1, 5, 4,  1> { static const int value =  256; };
template <> struct vnode_shift<1, 5, 4,  2> { static const int value =  0; };
// ILS = 5, CHECK_NODE = 5
template <> struct vnode_shift<1, 5, 5,  0> { static const int value =  219; };
template <> struct vnode_shift<1, 5, 5,  1> { static const int value =  130; };
template <> struct vnode_shift<1, 5, 5,  2> { static const int value =  251; };
template <> struct vnode_shift<1, 5, 5,  3> { static const int value =  322; };
template <> struct vnode_shift<1, 5, 5,  4> { static const int value =  295; };
template <> struct vnode_shift<1, 5, 5,  5> { static const int value =  258; };
template <> struct vnode_shift<1, 5, 5,  6> { static const int value =  283; };
template <> struct vnode_shift<1, 5, 5,  7> { static const int value =  0; };
// ILS = 5, CHECK_NODE = 6
template <> struct vnode_shift<1, 5, 6,  0> { static const int value =  294; };
template <> struct vnode_shift<1, 5, 6,  1> { static const int value =  73; };
template <> struct vnode_shift<1, 5, 6,  2> { static const int value =  330; };
template <> struct vnode_shift<1, 5, 6,  3> { static const int value =  99; };
template <> struct vnode_shift<1, 5, 6,  4> { static const int value =  172; };
template <> struct vnode_shift<1, 5, 6,  5> { static const int value =  150; };
template <> struct vnode_shift<1, 5, 6,  6> { static const int value =  284; };
template <> struct vnode_shift<1, 5, 6,  7> { static const int value =  305; };
template <> struct vnode_shift<1, 5, 6,  8> { static const int value =  0; };
// ILS = 5, CHECK_NODE = 7
template <> struct vnode_shift<1, 5, 7,  0> { static const int value =  3; };
template <> struct vnode_shift<1, 5, 7,  1> { static const int value =  103; };
template <> struct vnode_shift<1, 5, 7,  2> { static const int value =  224; };
template <> struct vnode_shift<1, 5, 7,  3> { static const int value =  297; };
template <> struct vnode_shift<1, 5, 7,  4> { static const int value =  215; };
template <> struct vnode_shift<1, 5, 7,  5> { static const int value =  39; };
template <> struct vnode_shift<1, 5, 7,  6> { static const int value =  0; };
// ILS = 5, CHECK_NODE = 8
template <> struct vnode_shift<1, 5, 8,  0> { static const int value =  348; };
template <> struct vnode_shift<1, 5, 8,  1> { static const int value =  75; };
template <> struct vnode_shift<1, 5, 8,  2> { static const int value =  22; };
template <> struct vnode_shift<1, 5, 8,  3> { static const int value =  312; };
template <> struct vnode_shift<1, 5, 8,  4> { static const int value =  224; };
template <> struct vnode_shift<1, 5, 8,  5> { static const int value =  17; };
template <> struct vnode_shift<1, 5, 8,  6> { static const int value =  59; };
template <> struct vnode_shift<1, 5, 8,  7> { static const int value =  314; };
template <> struct vnode_shift<1, 5, 8,  8> { static const int value =  244; };
template <> struct vnode_shift<1, 5, 8,  9> { static const int value =  0; };
// ILS = 5, CHECK_NODE = 9
template <> struct vnode_shift<1, 5, 9,  0> { static const int value =  156; };
template <> struct vnode_shift<1, 5, 9,  1> { static const int value =  88; };
template <> struct vnode_shift<1, 5, 9,  2> { static const int value =  293; };
template <> struct vnode_shift<1, 5, 9,  3> { static const int value =  111; };
template <> struct vnode_shift<1, 5, 9,  4> { static const int value =  92; };
template <> struct vnode_shift<1, 5, 9,  5> { static const int value =  152; };
template <> struct vnode_shift<1, 5, 9,  6> { static const int value =  23; };
template <> struct vnode_shift<1, 5, 9,  7> { static const int value =  337; };
template <> struct vnode_shift<1, 5, 9,  8> { static const int value =  0; };
// ILS = 5, CHECK_NODE = 10
template <> struct vnode_shift<1, 5, 10,  0> { static const int value =  175; };
template <> struct vnode_shift<1, 5, 10,  1> { static const int value =  253; };
template <> struct vnode_shift<1, 5, 10,  2> { static const int value =  27; };
template <> struct vnode_shift<1, 5, 10,  3> { static const int value =  231; };
template <> struct vnode_shift<1, 5, 10,  4> { static const int value =  49; };
template <> struct vnode_shift<1, 5, 10,  5> { static const int value =  267; };
template <> struct vnode_shift<1, 5, 10,  6> { static const int value =  0; };
// ILS = 5, CHECK_NODE = 11
template <> struct vnode_shift<1, 5, 11,  0> { static const int value =  25; };
template <> struct vnode_shift<1, 5, 11,  1> { static const int value =  322; };
template <> struct vnode_shift<1, 5, 11,  2> { static const int value =  200; };
template <> struct vnode_shift<1, 5, 11,  3> { static const int value =  351; };
template <> struct vnode_shift<1, 5, 11,  4> { static const int value =  166; };
template <> struct vnode_shift<1, 5, 11,  5> { static const int value =  338; };
template <> struct vnode_shift<1, 5, 11,  6> { static const int value =  192; };
template <> struct vnode_shift<1, 5, 11,  7> { static const int value =  0; };
// ILS = 5, CHECK_NODE = 12
template <> struct vnode_shift<1, 5, 12,  0> { static const int value =  123; };
template <> struct vnode_shift<1, 5, 12,  1> { static const int value =  217; };
template <> struct vnode_shift<1, 5, 12,  2> { static const int value =  142; };
template <> struct vnode_shift<1, 5, 12,  3> { static const int value =  110; };
template <> struct vnode_shift<1, 5, 12,  4> { static const int value =  176; };
template <> struct vnode_shift<1, 5, 12,  5> { static const int value =  76; };
template <> struct vnode_shift<1, 5, 12,  6> { static const int value =  0; };
// ILS = 5, CHECK_NODE = 13
template <> struct vnode_shift<1, 5, 13,  0> { static const int value =  311; };
template <> struct vnode_shift<1, 5, 13,  1> { static const int value =  251; };
template <> struct vnode_shift<1, 5, 13,  2> { static const int value =  265; };
template <> struct vnode_shift<1, 5, 13,  3> { static const int value =  94; };
template <> struct vnode_shift<1, 5, 13,  4> { static const int value =  81; };
template <> struct vnode_shift<1, 5, 13,  5> { static const int value =  0; };
// ILS = 5, CHECK_NODE = 14
template <> struct vnode_shift<1, 5, 14,  0> { static const int value =  22; };
template <> struct vnode_shift<1, 5, 14,  1> { static const int value =  322; };
template <> struct vnode_shift<1, 5, 14,  2> { static const int value =  277; };
template <> struct vnode_shift<1, 5, 14,  3> { static const int value =  156; };
template <> struct vnode_shift<1, 5, 14,  4> { static const int value =  66; };
template <> struct vnode_shift<1, 5, 14,  5> { static const int value =  78; };
template <> struct vnode_shift<1, 5, 14,  6> { static const int value =  0; };
// ILS = 5, CHECK_NODE = 15
template <> struct vnode_shift<1, 5, 15,  0> { static const int value =  176; };
template <> struct vnode_shift<1, 5, 15,  1> { static const int value =  348; };
template <> struct vnode_shift<1, 5, 15,  2> { static const int value =  15; };
template <> struct vnode_shift<1, 5, 15,  3> { static const int value =  81; };
template <> struct vnode_shift<1, 5, 15,  4> { static const int value =  176; };
template <> struct vnode_shift<1, 5, 15,  5> { static const int value =  113; };
template <> struct vnode_shift<1, 5, 15,  6> { static const int value =  0; };
// ILS = 5, CHECK_NODE = 16
template <> struct vnode_shift<1, 5, 16,  0> { static const int value =  190; };
template <> struct vnode_shift<1, 5, 16,  1> { static const int value =  293; };
template <> struct vnode_shift<1, 5, 16,  2> { static const int value =  332; };
template <> struct vnode_shift<1, 5, 16,  3> { static const int value =  331; };
template <> struct vnode_shift<1, 5, 16,  4> { static const int value =  114; };
template <> struct vnode_shift<1, 5, 16,  5> { static const int value =  0; };
// ILS = 5, CHECK_NODE = 17
template <> struct vnode_shift<1, 5, 17,  0> { static const int value =  110; };
template <> struct vnode_shift<1, 5, 17,  1> { static const int value =  228; };
template <> struct vnode_shift<1, 5, 17,  2> { static const int value =  247; };
template <> struct vnode_shift<1, 5, 17,  3> { static const int value =  116; };
template <> struct vnode_shift<1, 5, 17,  4> { static const int value =  190; };
template <> struct vnode_shift<1, 5, 17,  5> { static const int value =  0; };
// ILS = 5, CHECK_NODE = 18
template <> struct vnode_shift<1, 5, 18,  0> { static const int value =  47; };
template <> struct vnode_shift<1, 5, 18,  1> { static const int value =  286; };
template <> struct vnode_shift<1, 5, 18,  2> { static const int value =  246; };
template <> struct vnode_shift<1, 5, 18,  3> { static const int value =  181; };
template <> struct vnode_shift<1, 5, 18,  4> { static const int value =  73; };
template <> struct vnode_shift<1, 5, 18,  5> { static const int value =  0; };
// ILS = 5, CHECK_NODE = 19
template <> struct vnode_shift<1, 5, 19,  0> { static const int value =  87; };
template <> struct vnode_shift<1, 5, 19,  1> { static const int value =  110; };
template <> struct vnode_shift<1, 5, 19,  2> { static const int value =  147; };
template <> struct vnode_shift<1, 5, 19,  3> { static const int value =  258; };
template <> struct vnode_shift<1, 5, 19,  4> { static const int value =  204; };
template <> struct vnode_shift<1, 5, 19,  5> { static const int value =  0; };
// ILS = 5, CHECK_NODE = 20
template <> struct vnode_shift<1, 5, 20,  0> { static const int value =  89; };
template <> struct vnode_shift<1, 5, 20,  1> { static const int value =  65; };
template <> struct vnode_shift<1, 5, 20,  2> { static const int value =  155; };
template <> struct vnode_shift<1, 5, 20,  3> { static const int value =  244; };
template <> struct vnode_shift<1, 5, 20,  4> { static const int value =  30; };
template <> struct vnode_shift<1, 5, 20,  5> { static const int value =  0; };
// ILS = 5, CHECK_NODE = 21
template <> struct vnode_shift<1, 5, 21,  0> { static const int value =  162; };
template <> struct vnode_shift<1, 5, 21,  1> { static const int value =  264; };
template <> struct vnode_shift<1, 5, 21,  2> { static const int value =  346; };
template <> struct vnode_shift<1, 5, 21,  3> { static const int value =  143; };
template <> struct vnode_shift<1, 5, 21,  4> { static const int value =  109; };
template <> struct vnode_shift<1, 5, 21,  5> { static const int value =  0; };
// ILS = 5, CHECK_NODE = 22
template <> struct vnode_shift<1, 5, 22,  0> { static const int value =  280; };
template <> struct vnode_shift<1, 5, 22,  1> { static const int value =  157; };
template <> struct vnode_shift<1, 5, 22,  2> { static const int value =  236; };
template <> struct vnode_shift<1, 5, 22,  3> { static const int value =  113; };
template <> struct vnode_shift<1, 5, 22,  4> { static const int value =  0; };
// ILS = 5, CHECK_NODE = 23
template <> struct vnode_shift<1, 5, 23,  0> { static const int value =  18; };
template <> struct vnode_shift<1, 5, 23,  1> { static const int value =  6; };
template <> struct vnode_shift<1, 5, 23,  2> { static const int value =  181; };
template <> struct vnode_shift<1, 5, 23,  3> { static const int value =  304; };
template <> struct vnode_shift<1, 5, 23,  4> { static const int value =  0; };
// ILS = 5, CHECK_NODE = 24
template <> struct vnode_shift<1, 5, 24,  0> { static const int value =  38; };
template <> struct vnode_shift<1, 5, 24,  1> { static const int value =  170; };
template <> struct vnode_shift<1, 5, 24,  2> { static const int value =  249; };
template <> struct vnode_shift<1, 5, 24,  3> { static const int value =  288; };
template <> struct vnode_shift<1, 5, 24,  4> { static const int value =  194; };
template <> struct vnode_shift<1, 5, 24,  5> { static const int value =  0; };
// ILS = 5, CHECK_NODE = 25
template <> struct vnode_shift<1, 5, 25,  0> { static const int value =  279; };
template <> struct vnode_shift<1, 5, 25,  1> { static const int value =  255; };
template <> struct vnode_shift<1, 5, 25,  2> { static const int value =  111; };
template <> struct vnode_shift<1, 5, 25,  3> { static const int value =  54; };
template <> struct vnode_shift<1, 5, 25,  4> { static const int value =  0; };
// ILS = 5, CHECK_NODE = 26
template <> struct vnode_shift<1, 5, 26,  0> { static const int value =  325; };
template <> struct vnode_shift<1, 5, 26,  1> { static const int value =  326; };
template <> struct vnode_shift<1, 5, 26,  2> { static const int value =  226; };
template <> struct vnode_shift<1, 5, 26,  3> { static const int value =  99; };
template <> struct vnode_shift<1, 5, 26,  4> { static const int value =  0; };
// ILS = 5, CHECK_NODE = 27
template <> struct vnode_shift<1, 5, 27,  0> { static const int value =  91; };
template <> struct vnode_shift<1, 5, 27,  1> { static const int value =  326; };
template <> struct vnode_shift<1, 5, 27,  2> { static const int value =  268; };
template <> struct vnode_shift<1, 5, 27,  3> { static const int value =  0; };
// ILS = 5, CHECK_NODE = 28
template <> struct vnode_shift<1, 5, 28,  0> { static const int value =  102; };
template <> struct vnode_shift<1, 5, 28,  1> { static const int value =  1; };
template <> struct vnode_shift<1, 5, 28,  2> { static const int value =  40; };
template <> struct vnode_shift<1, 5, 28,  3> { static const int value =  167; };
template <> struct vnode_shift<1, 5, 28,  4> { static const int value =  0; };
// ILS = 5, CHECK_NODE = 29
template <> struct vnode_shift<1, 5, 29,  0> { static const int value =  273; };
template <> struct vnode_shift<1, 5, 29,  1> { static const int value =  104; };
template <> struct vnode_shift<1, 5, 29,  2> { static const int value =  243; };
template <> struct vnode_shift<1, 5, 29,  3> { static const int value =  107; };
template <> struct vnode_shift<1, 5, 29,  4> { static const int value =  0; };
// ILS = 5, CHECK_NODE = 30
template <> struct vnode_shift<1, 5, 30,  0> { static const int value =  171; };
template <> struct vnode_shift<1, 5, 30,  1> { static const int value =  16; };
template <> struct vnode_shift<1, 5, 30,  2> { static const int value =  95; };
template <> struct vnode_shift<1, 5, 30,  3> { static const int value =  212; };
template <> struct vnode_shift<1, 5, 30,  4> { static const int value =  0; };
// ILS = 5, CHECK_NODE = 31
template <> struct vnode_shift<1, 5, 31,  0> { static const int value =  101; };
template <> struct vnode_shift<1, 5, 31,  1> { static const int value =  297; };
template <> struct vnode_shift<1, 5, 31,  2> { static const int value =  279; };
template <> struct vnode_shift<1, 5, 31,  3> { static const int value =  222; };
template <> struct vnode_shift<1, 5, 31,  4> { static const int value =  0; };
// ILS = 5, CHECK_NODE = 32
template <> struct vnode_shift<1, 5, 32,  0> { static const int value =  351; };
template <> struct vnode_shift<1, 5, 32,  1> { static const int value =  265; };
template <> struct vnode_shift<1, 5, 32,  2> { static const int value =  338; };
template <> struct vnode_shift<1, 5, 32,  3> { static const int value =  83; };
template <> struct vnode_shift<1, 5, 32,  4> { static const int value =  0; };
// ILS = 5, CHECK_NODE = 33
template <> struct vnode_shift<1, 5, 33,  0> { static const int value =  56; };
template <> struct vnode_shift<1, 5, 33,  1> { static const int value =  304; };
template <> struct vnode_shift<1, 5, 33,  2> { static const int value =  141; };
template <> struct vnode_shift<1, 5, 33,  3> { static const int value =  101; };
template <> struct vnode_shift<1, 5, 33,  4> { static const int value =  0; };
// ILS = 5, CHECK_NODE = 34
template <> struct vnode_shift<1, 5, 34,  0> { static const int value =  60; };
template <> struct vnode_shift<1, 5, 34,  1> { static const int value =  320; };
template <> struct vnode_shift<1, 5, 34,  2> { static const int value =  112; };
template <> struct vnode_shift<1, 5, 34,  3> { static const int value =  54; };
template <> struct vnode_shift<1, 5, 34,  4> { static const int value =  0; };
// ILS = 5, CHECK_NODE = 35
template <> struct vnode_shift<1, 5, 35,  0> { static const int value =  100; };
template <> struct vnode_shift<1, 5, 35,  1> { static const int value =  210; };
template <> struct vnode_shift<1, 5, 35,  2> { static const int value =  195; };
template <> struct vnode_shift<1, 5, 35,  3> { static const int value =  268; };
template <> struct vnode_shift<1, 5, 35,  4> { static const int value =  0; };
// ILS = 5, CHECK_NODE = 36
template <> struct vnode_shift<1, 5, 36,  0> { static const int value =  135; };
template <> struct vnode_shift<1, 5, 36,  1> { static const int value =  15; };
template <> struct vnode_shift<1, 5, 36,  2> { static const int value =  35; };
template <> struct vnode_shift<1, 5, 36,  3> { static const int value =  188; };
template <> struct vnode_shift<1, 5, 36,  4> { static const int value =  0; };
// ILS = 5, CHECK_NODE = 37
template <> struct vnode_shift<1, 5, 37,  0> { static const int value =  319; };
template <> struct vnode_shift<1, 5, 37,  1> { static const int value =  236; };
template <> struct vnode_shift<1, 5, 37,  2> { static const int value =  85; };
template <> struct vnode_shift<1, 5, 37,  3> { static const int value =  0; };
// ILS = 5, CHECK_NODE = 38
template <> struct vnode_shift<1, 5, 38,  0> { static const int value =  164; };
template <> struct vnode_shift<1, 5, 38,  1> { static const int value =  196; };
template <> struct vnode_shift<1, 5, 38,  2> { static const int value =  209; };
template <> struct vnode_shift<1, 5, 38,  3> { static const int value =  246; };
template <> struct vnode_shift<1, 5, 38,  4> { static const int value =  0; };
// ILS = 5, CHECK_NODE = 39
template <> struct vnode_shift<1, 5, 39,  0> { static const int value =  236; };
template <> struct vnode_shift<1, 5, 39,  1> { static const int value =  264; };
template <> struct vnode_shift<1, 5, 39,  2> { static const int value =  37; };
template <> struct vnode_shift<1, 5, 39,  3> { static const int value =  272; };
template <> struct vnode_shift<1, 5, 39,  4> { static const int value =  0; };
// ILS = 5, CHECK_NODE = 40
template <> struct vnode_shift<1, 5, 40,  0> { static const int value =  304; };
template <> struct vnode_shift<1, 5, 40,  1> { static const int value =  237; };
template <> struct vnode_shift<1, 5, 40,  2> { static const int value =  135; };
template <> struct vnode_shift<1, 5, 40,  3> { static const int value =  0; };
// ILS = 5, CHECK_NODE = 41
template <> struct vnode_shift<1, 5, 41,  0> { static const int value =  123; };
template <> struct vnode_shift<1, 5, 41,  1> { static const int value =  77; };
template <> struct vnode_shift<1, 5, 41,  2> { static const int value =  25; };
template <> struct vnode_shift<1, 5, 41,  3> { static const int value =  272; };
template <> struct vnode_shift<1, 5, 41,  4> { static const int value =  0; };
// ILS = 5, CHECK_NODE = 42
template <> struct vnode_shift<1, 5, 42,  0> { static const int value =  288; };
template <> struct vnode_shift<1, 5, 42,  1> { static const int value =  83; };
template <> struct vnode_shift<1, 5, 42,  2> { static const int value =  17; };
template <> struct vnode_shift<1, 5, 42,  3> { static const int value =  0; };
// ILS = 5, CHECK_NODE = 43
template <> struct vnode_shift<1, 5, 43,  0> { static const int value =  210; };
template <> struct vnode_shift<1, 5, 43,  1> { static const int value =  3; };
template <> struct vnode_shift<1, 5, 43,  2> { static const int value =  53; };
template <> struct vnode_shift<1, 5, 43,  3> { static const int value =  167; };
template <> struct vnode_shift<1, 5, 43,  4> { static const int value =  0; };
// ILS = 5, CHECK_NODE = 44
template <> struct vnode_shift<1, 5, 44,  0> { static const int value =  79; };
template <> struct vnode_shift<1, 5, 44,  1> { static const int value =  244; };
template <> struct vnode_shift<1, 5, 44,  2> { static const int value =  293; };
template <> struct vnode_shift<1, 5, 44,  3> { static const int value =  272; };
template <> struct vnode_shift<1, 5, 44,  4> { static const int value =  0; };
// ILS = 5, CHECK_NODE = 45
template <> struct vnode_shift<1, 5, 45,  0> { static const int value =  82; };
template <> struct vnode_shift<1, 5, 45,  1> { static const int value =  67; };
template <> struct vnode_shift<1, 5, 45,  2> { static const int value =  235; };
template <> struct vnode_shift<1, 5, 45,  3> { static const int value =  0; };
// ILS = 6, CHECK_NODE = 0
template <> struct vnode_shift<1, 6, 0,  0> { static const int value =  0; };
template <> struct vnode_shift<1, 6, 0,  1> { static const int value =  0; };
template <> struct vnode_shift<1, 6, 0,  2> { static const int value =  0; };
template <> struct vnode_shift<1, 6, 0,  3> { static const int value =  0; };
template <> struct vnode_shift<1, 6, 0,  4> { static const int value =  0; };
template <> struct vnode_shift<1, 6, 0,  5> { static const int value =  0; };
template <> struct vnode_shift<1, 6, 0,  6> { static const int value =  0; };
template <> struct vnode_shift<1, 6, 0,  7> { static const int value =  0; };
template <> struct vnode_shift<1, 6, 0,  8> { static const int value =  0; };
template <> struct vnode_shift<1, 6, 0,  9> { static const int value =  0; };
template <> struct vnode_shift<1, 6, 0,  10> { static const int value =  0; };
template <> struct vnode_shift<1, 6, 0,  11> { static const int value =  0; };
template <> struct vnode_shift<1, 6, 0,  12> { static const int value =  0; };
template <> struct vnode_shift<1, 6, 0,  13> { static const int value =  0; };
template <> struct vnode_shift<1, 6, 0,  14> { static const int value =  0; };
template <> struct vnode_shift<1, 6, 0,  15> { static const int value =  0; };
template <> struct vnode_shift<1, 6, 0,  16> { static const int value =  0; };
template <> struct vnode_shift<1, 6, 0,  17> { static const int value =  0; };
template <> struct vnode_shift<1, 6, 0,  18> { static const int value =  0; };
// ILS = 6, CHECK_NODE = 1
template <> struct vnode_shift<1, 6, 1,  0> { static const int value =  22; };
template <> struct vnode_shift<1, 6, 1,  1> { static const int value =  11; };
template <> struct vnode_shift<1, 6, 1,  2> { static const int value =  124; };
template <> struct vnode_shift<1, 6, 1,  3> { static const int value =  0; };
template <> struct vnode_shift<1, 6, 1,  4> { static const int value =  10; };
template <> struct vnode_shift<1, 6, 1,  5> { static const int value =  0; };
template <> struct vnode_shift<1, 6, 1,  6> { static const int value =  0; };
template <> struct vnode_shift<1, 6, 1,  7> { static const int value =  2; };
template <> struct vnode_shift<1, 6, 1,  8> { static const int value =  16; };
template <> struct vnode_shift<1, 6, 1,  9> { static const int value =  60; };
template <> struct vnode_shift<1, 6, 1,  10> { static const int value =  0; };
template <> struct vnode_shift<1, 6, 1,  11> { static const int value =  6; };
template <> struct vnode_shift<1, 6, 1,  12> { static const int value =  30; };
template <> struct vnode_shift<1, 6, 1,  13> { static const int value =  0; };
template <> struct vnode_shift<1, 6, 1,  14> { static const int value =  168; };
template <> struct vnode_shift<1, 6, 1,  15> { static const int value =  31; };
template <> struct vnode_shift<1, 6, 1,  16> { static const int value =  105; };
template <> struct vnode_shift<1, 6, 1,  17> { static const int value =  0; };
template <> struct vnode_shift<1, 6, 1,  18> { static const int value =  0; };
// ILS = 6, CHECK_NODE = 2
template <> struct vnode_shift<1, 6, 2,  0> { static const int value =  132; };
template <> struct vnode_shift<1, 6, 2,  1> { static const int value =  37; };
template <> struct vnode_shift<1, 6, 2,  2> { static const int value =  21; };
template <> struct vnode_shift<1, 6, 2,  3> { static const int value =  180; };
template <> struct vnode_shift<1, 6, 2,  4> { static const int value =  4; };
template <> struct vnode_shift<1, 6, 2,  5> { static const int value =  149; };
template <> struct vnode_shift<1, 6, 2,  6> { static const int value =  48; };
template <> struct vnode_shift<1, 6, 2,  7> { static const int value =  38; };
template <> struct vnode_shift<1, 6, 2,  8> { static const int value =  122; };
template <> struct vnode_shift<1, 6, 2,  9> { static const int value =  195; };
template <> struct vnode_shift<1, 6, 2,  10> { static const int value =  155; };
template <> struct vnode_shift<1, 6, 2,  11> { static const int value =  28; };
template <> struct vnode_shift<1, 6, 2,  12> { static const int value =  85; };
template <> struct vnode_shift<1, 6, 2,  13> { static const int value =  47; };
template <> struct vnode_shift<1, 6, 2,  14> { static const int value =  179; };
template <> struct vnode_shift<1, 6, 2,  15> { static const int value =  42; };
template <> struct vnode_shift<1, 6, 2,  16> { static const int value =  66; };
template <> struct vnode_shift<1, 6, 2,  17> { static const int value =  0; };
template <> struct vnode_shift<1, 6, 2,  18> { static const int value =  0; };
// ILS = 6, CHECK_NODE = 3
template <> struct vnode_shift<1, 6, 3,  0> { static const int value =  4; };
template <> struct vnode_shift<1, 6, 3,  1> { static const int value =  6; };
template <> struct vnode_shift<1, 6, 3,  2> { static const int value =  33; };
template <> struct vnode_shift<1, 6, 3,  3> { static const int value =  113; };
template <> struct vnode_shift<1, 6, 3,  4> { static const int value =  49; };
template <> struct vnode_shift<1, 6, 3,  5> { static const int value =  21; };
template <> struct vnode_shift<1, 6, 3,  6> { static const int value =  6; };
template <> struct vnode_shift<1, 6, 3,  7> { static const int value =  151; };
template <> struct vnode_shift<1, 6, 3,  8> { static const int value =  83; };
template <> struct vnode_shift<1, 6, 3,  9> { static const int value =  154; };
template <> struct vnode_shift<1, 6, 3,  10> { static const int value =  87; };
template <> struct vnode_shift<1, 6, 3,  11> { static const int value =  5; };
template <> struct vnode_shift<1, 6, 3,  12> { static const int value =  92; };
template <> struct vnode_shift<1, 6, 3,  13> { static const int value =  173; };
template <> struct vnode_shift<1, 6, 3,  14> { static const int value =  120; };
template <> struct vnode_shift<1, 6, 3,  15> { static const int value =  2; };
template <> struct vnode_shift<1, 6, 3,  16> { static const int value =  142; };
template <> struct vnode_shift<1, 6, 3,  17> { static const int value =  0; };
template <> struct vnode_shift<1, 6, 3,  18> { static const int value =  0; };
// ILS = 6, CHECK_NODE = 4
template <> struct vnode_shift<1, 6, 4,  0> { static const int value =  24; };
template <> struct vnode_shift<1, 6, 4,  1> { static const int value =  204; };
template <> struct vnode_shift<1, 6, 4,  2> { static const int value =  0; };
// ILS = 6, CHECK_NODE = 5
template <> struct vnode_shift<1, 6, 5,  0> { static const int value =  185; };
template <> struct vnode_shift<1, 6, 5,  1> { static const int value =  100; };
template <> struct vnode_shift<1, 6, 5,  2> { static const int value =  24; };
template <> struct vnode_shift<1, 6, 5,  3> { static const int value =  65; };
template <> struct vnode_shift<1, 6, 5,  4> { static const int value =  207; };
template <> struct vnode_shift<1, 6, 5,  5> { static const int value =  161; };
template <> struct vnode_shift<1, 6, 5,  6> { static const int value =  72; };
template <> struct vnode_shift<1, 6, 5,  7> { static const int value =  0; };
// ILS = 6, CHECK_NODE = 6
template <> struct vnode_shift<1, 6, 6,  0> { static const int value =  6; };
template <> struct vnode_shift<1, 6, 6,  1> { static const int value =  27; };
template <> struct vnode_shift<1, 6, 6,  2> { static const int value =  163; };
template <> struct vnode_shift<1, 6, 6,  3> { static const int value =  50; };
template <> struct vnode_shift<1, 6, 6,  4> { static const int value =  48; };
template <> struct vnode_shift<1, 6, 6,  5> { static const int value =  24; };
template <> struct vnode_shift<1, 6, 6,  6> { static const int value =  38; };
template <> struct vnode_shift<1, 6, 6,  7> { static const int value =  91; };
template <> struct vnode_shift<1, 6, 6,  8> { static const int value =  0; };
// ILS = 6, CHECK_NODE = 7
template <> struct vnode_shift<1, 6, 7,  0> { static const int value =  145; };
template <> struct vnode_shift<1, 6, 7,  1> { static const int value =  88; };
template <> struct vnode_shift<1, 6, 7,  2> { static const int value =  112; };
template <> struct vnode_shift<1, 6, 7,  3> { static const int value =  153; };
template <> struct vnode_shift<1, 6, 7,  4> { static const int value =  159; };
template <> struct vnode_shift<1, 6, 7,  5> { static const int value =  76; };
template <> struct vnode_shift<1, 6, 7,  6> { static const int value =  0; };
// ILS = 6, CHECK_NODE = 8
template <> struct vnode_shift<1, 6, 8,  0> { static const int value =  172; };
template <> struct vnode_shift<1, 6, 8,  1> { static const int value =  2; };
template <> struct vnode_shift<1, 6, 8,  2> { static const int value =  131; };
template <> struct vnode_shift<1, 6, 8,  3> { static const int value =  141; };
template <> struct vnode_shift<1, 6, 8,  4> { static const int value =  96; };
template <> struct vnode_shift<1, 6, 8,  5> { static const int value =  99; };
template <> struct vnode_shift<1, 6, 8,  6> { static const int value =  101; };
template <> struct vnode_shift<1, 6, 8,  7> { static const int value =  35; };
template <> struct vnode_shift<1, 6, 8,  8> { static const int value =  116; };
template <> struct vnode_shift<1, 6, 8,  9> { static const int value =  0; };
// ILS = 6, CHECK_NODE = 9
template <> struct vnode_shift<1, 6, 9,  0> { static const int value =  6; };
template <> struct vnode_shift<1, 6, 9,  1> { static const int value =  10; };
template <> struct vnode_shift<1, 6, 9,  2> { static const int value =  145; };
template <> struct vnode_shift<1, 6, 9,  3> { static const int value =  53; };
template <> struct vnode_shift<1, 6, 9,  4> { static const int value =  201; };
template <> struct vnode_shift<1, 6, 9,  5> { static const int value =  4; };
template <> struct vnode_shift<1, 6, 9,  6> { static const int value =  164; };
template <> struct vnode_shift<1, 6, 9,  7> { static const int value =  173; };
template <> struct vnode_shift<1, 6, 9,  8> { static const int value =  0; };
// ILS = 6, CHECK_NODE = 10
template <> struct vnode_shift<1, 6, 10,  0> { static const int value =  126; };
template <> struct vnode_shift<1, 6, 10,  1> { static const int value =  77; };
template <> struct vnode_shift<1, 6, 10,  2> { static const int value =  156; };
template <> struct vnode_shift<1, 6, 10,  3> { static const int value =  16; };
template <> struct vnode_shift<1, 6, 10,  4> { static const int value =  12; };
template <> struct vnode_shift<1, 6, 10,  5> { static const int value =  70; };
template <> struct vnode_shift<1, 6, 10,  6> { static const int value =  0; };
// ILS = 6, CHECK_NODE = 11
template <> struct vnode_shift<1, 6, 11,  0> { static const int value =  184; };
template <> struct vnode_shift<1, 6, 11,  1> { static const int value =  194; };
template <> struct vnode_shift<1, 6, 11,  2> { static const int value =  123; };
template <> struct vnode_shift<1, 6, 11,  3> { static const int value =  16; };
template <> struct vnode_shift<1, 6, 11,  4> { static const int value =  104; };
template <> struct vnode_shift<1, 6, 11,  5> { static const int value =  109; };
template <> struct vnode_shift<1, 6, 11,  6> { static const int value =  124; };
template <> struct vnode_shift<1, 6, 11,  7> { static const int value =  0; };
// ILS = 6, CHECK_NODE = 12
template <> struct vnode_shift<1, 6, 12,  0> { static const int value =  6; };
template <> struct vnode_shift<1, 6, 12,  1> { static const int value =  20; };
template <> struct vnode_shift<1, 6, 12,  2> { static const int value =  203; };
template <> struct vnode_shift<1, 6, 12,  3> { static const int value =  153; };
template <> struct vnode_shift<1, 6, 12,  4> { static const int value =  104; };
template <> struct vnode_shift<1, 6, 12,  5> { static const int value =  207; };
template <> struct vnode_shift<1, 6, 12,  6> { static const int value =  0; };
// ILS = 6, CHECK_NODE = 13
template <> struct vnode_shift<1, 6, 13,  0> { static const int value =  52; };
template <> struct vnode_shift<1, 6, 13,  1> { static const int value =  147; };
template <> struct vnode_shift<1, 6, 13,  2> { static const int value =  1; };
template <> struct vnode_shift<1, 6, 13,  3> { static const int value =  16; };
template <> struct vnode_shift<1, 6, 13,  4> { static const int value =  46; };
template <> struct vnode_shift<1, 6, 13,  5> { static const int value =  0; };
// ILS = 6, CHECK_NODE = 14
template <> struct vnode_shift<1, 6, 14,  0> { static const int value =  1; };
template <> struct vnode_shift<1, 6, 14,  1> { static const int value =  202; };
template <> struct vnode_shift<1, 6, 14,  2> { static const int value =  118; };
template <> struct vnode_shift<1, 6, 14,  3> { static const int value =  130; };
template <> struct vnode_shift<1, 6, 14,  4> { static const int value =  1; };
template <> struct vnode_shift<1, 6, 14,  5> { static const int value =  2; };
template <> struct vnode_shift<1, 6, 14,  6> { static const int value =  0; };
// ILS = 6, CHECK_NODE = 15
template <> struct vnode_shift<1, 6, 15,  0> { static const int value =  173; };
template <> struct vnode_shift<1, 6, 15,  1> { static const int value =  6; };
template <> struct vnode_shift<1, 6, 15,  2> { static const int value =  81; };
template <> struct vnode_shift<1, 6, 15,  3> { static const int value =  182; };
template <> struct vnode_shift<1, 6, 15,  4> { static const int value =  53; };
template <> struct vnode_shift<1, 6, 15,  5> { static const int value =  46; };
template <> struct vnode_shift<1, 6, 15,  6> { static const int value =  0; };
// ILS = 6, CHECK_NODE = 16
template <> struct vnode_shift<1, 6, 16,  0> { static const int value =  88; };
template <> struct vnode_shift<1, 6, 16,  1> { static const int value =  198; };
template <> struct vnode_shift<1, 6, 16,  2> { static const int value =  160; };
template <> struct vnode_shift<1, 6, 16,  3> { static const int value =  122; };
template <> struct vnode_shift<1, 6, 16,  4> { static const int value =  182; };
template <> struct vnode_shift<1, 6, 16,  5> { static const int value =  0; };
// ILS = 6, CHECK_NODE = 17
template <> struct vnode_shift<1, 6, 17,  0> { static const int value =  91; };
template <> struct vnode_shift<1, 6, 17,  1> { static const int value =  184; };
template <> struct vnode_shift<1, 6, 17,  2> { static const int value =  30; };
template <> struct vnode_shift<1, 6, 17,  3> { static const int value =  3; };
template <> struct vnode_shift<1, 6, 17,  4> { static const int value =  155; };
template <> struct vnode_shift<1, 6, 17,  5> { static const int value =  0; };
// ILS = 6, CHECK_NODE = 18
template <> struct vnode_shift<1, 6, 18,  0> { static const int value =  1; };
template <> struct vnode_shift<1, 6, 18,  1> { static const int value =  41; };
template <> struct vnode_shift<1, 6, 18,  2> { static const int value =  167; };
template <> struct vnode_shift<1, 6, 18,  3> { static const int value =  68; };
template <> struct vnode_shift<1, 6, 18,  4> { static const int value =  148; };
template <> struct vnode_shift<1, 6, 18,  5> { static const int value =  0; };
// ILS = 6, CHECK_NODE = 19
template <> struct vnode_shift<1, 6, 19,  0> { static const int value =  12; };
template <> struct vnode_shift<1, 6, 19,  1> { static const int value =  6; };
template <> struct vnode_shift<1, 6, 19,  2> { static const int value =  166; };
template <> struct vnode_shift<1, 6, 19,  3> { static const int value =  184; };
template <> struct vnode_shift<1, 6, 19,  4> { static const int value =  191; };
template <> struct vnode_shift<1, 6, 19,  5> { static const int value =  0; };
// ILS = 6, CHECK_NODE = 20
template <> struct vnode_shift<1, 6, 20,  0> { static const int value =  6; };
template <> struct vnode_shift<1, 6, 20,  1> { static const int value =  12; };
template <> struct vnode_shift<1, 6, 20,  2> { static const int value =  15; };
template <> struct vnode_shift<1, 6, 20,  3> { static const int value =  5; };
template <> struct vnode_shift<1, 6, 20,  4> { static const int value =  30; };
template <> struct vnode_shift<1, 6, 20,  5> { static const int value =  0; };
// ILS = 6, CHECK_NODE = 21
template <> struct vnode_shift<1, 6, 21,  0> { static const int value =  6; };
template <> struct vnode_shift<1, 6, 21,  1> { static const int value =  86; };
template <> struct vnode_shift<1, 6, 21,  2> { static const int value =  96; };
template <> struct vnode_shift<1, 6, 21,  3> { static const int value =  42; };
template <> struct vnode_shift<1, 6, 21,  4> { static const int value =  199; };
template <> struct vnode_shift<1, 6, 21,  5> { static const int value =  0; };
// ILS = 6, CHECK_NODE = 22
template <> struct vnode_shift<1, 6, 22,  0> { static const int value =  44; };
template <> struct vnode_shift<1, 6, 22,  1> { static const int value =  58; };
template <> struct vnode_shift<1, 6, 22,  2> { static const int value =  130; };
template <> struct vnode_shift<1, 6, 22,  3> { static const int value =  131; };
template <> struct vnode_shift<1, 6, 22,  4> { static const int value =  0; };
// ILS = 6, CHECK_NODE = 23
template <> struct vnode_shift<1, 6, 23,  0> { static const int value =  45; };
template <> struct vnode_shift<1, 6, 23,  1> { static const int value =  18; };
template <> struct vnode_shift<1, 6, 23,  2> { static const int value =  132; };
template <> struct vnode_shift<1, 6, 23,  3> { static const int value =  100; };
template <> struct vnode_shift<1, 6, 23,  4> { static const int value =  0; };
// ILS = 6, CHECK_NODE = 24
template <> struct vnode_shift<1, 6, 24,  0> { static const int value =  9; };
template <> struct vnode_shift<1, 6, 24,  1> { static const int value =  125; };
template <> struct vnode_shift<1, 6, 24,  2> { static const int value =  191; };
template <> struct vnode_shift<1, 6, 24,  3> { static const int value =  28; };
template <> struct vnode_shift<1, 6, 24,  4> { static const int value =  6; };
template <> struct vnode_shift<1, 6, 24,  5> { static const int value =  0; };
// ILS = 6, CHECK_NODE = 25
template <> struct vnode_shift<1, 6, 25,  0> { static const int value =  4; };
template <> struct vnode_shift<1, 6, 25,  1> { static const int value =  74; };
template <> struct vnode_shift<1, 6, 25,  2> { static const int value =  16; };
template <> struct vnode_shift<1, 6, 25,  3> { static const int value =  28; };
template <> struct vnode_shift<1, 6, 25,  4> { static const int value =  0; };
// ILS = 6, CHECK_NODE = 26
template <> struct vnode_shift<1, 6, 26,  0> { static const int value =  21; };
template <> struct vnode_shift<1, 6, 26,  1> { static const int value =  142; };
template <> struct vnode_shift<1, 6, 26,  2> { static const int value =  192; };
template <> struct vnode_shift<1, 6, 26,  3> { static const int value =  197; };
template <> struct vnode_shift<1, 6, 26,  4> { static const int value =  0; };
// ILS = 6, CHECK_NODE = 27
template <> struct vnode_shift<1, 6, 27,  0> { static const int value =  98; };
template <> struct vnode_shift<1, 6, 27,  1> { static const int value =  140; };
template <> struct vnode_shift<1, 6, 27,  2> { static const int value =  22; };
template <> struct vnode_shift<1, 6, 27,  3> { static const int value =  0; };
// ILS = 6, CHECK_NODE = 28
template <> struct vnode_shift<1, 6, 28,  0> { static const int value =  4; };
template <> struct vnode_shift<1, 6, 28,  1> { static const int value =  1; };
template <> struct vnode_shift<1, 6, 28,  2> { static const int value =  40; };
template <> struct vnode_shift<1, 6, 28,  3> { static const int value =  93; };
template <> struct vnode_shift<1, 6, 28,  4> { static const int value =  0; };
// ILS = 6, CHECK_NODE = 29
template <> struct vnode_shift<1, 6, 29,  0> { static const int value =  92; };
template <> struct vnode_shift<1, 6, 29,  1> { static const int value =  136; };
template <> struct vnode_shift<1, 6, 29,  2> { static const int value =  106; };
template <> struct vnode_shift<1, 6, 29,  3> { static const int value =  6; };
template <> struct vnode_shift<1, 6, 29,  4> { static const int value =  0; };
// ILS = 6, CHECK_NODE = 30
template <> struct vnode_shift<1, 6, 30,  0> { static const int value =  2; };
template <> struct vnode_shift<1, 6, 30,  1> { static const int value =  88; };
template <> struct vnode_shift<1, 6, 30,  2> { static const int value =  112; };
template <> struct vnode_shift<1, 6, 30,  3> { static const int value =  20; };
template <> struct vnode_shift<1, 6, 30,  4> { static const int value =  0; };
// ILS = 6, CHECK_NODE = 31
template <> struct vnode_shift<1, 6, 31,  0> { static const int value =  4; };
template <> struct vnode_shift<1, 6, 31,  1> { static const int value =  49; };
template <> struct vnode_shift<1, 6, 31,  2> { static const int value =  125; };
template <> struct vnode_shift<1, 6, 31,  3> { static const int value =  194; };
template <> struct vnode_shift<1, 6, 31,  4> { static const int value =  0; };
// ILS = 6, CHECK_NODE = 32
template <> struct vnode_shift<1, 6, 32,  0> { static const int value =  6; };
template <> struct vnode_shift<1, 6, 32,  1> { static const int value =  126; };
template <> struct vnode_shift<1, 6, 32,  2> { static const int value =  63; };
template <> struct vnode_shift<1, 6, 32,  3> { static const int value =  20; };
template <> struct vnode_shift<1, 6, 32,  4> { static const int value =  0; };
// ILS = 6, CHECK_NODE = 33
template <> struct vnode_shift<1, 6, 33,  0> { static const int value =  10; };
template <> struct vnode_shift<1, 6, 33,  1> { static const int value =  30; };
template <> struct vnode_shift<1, 6, 33,  2> { static const int value =  6; };
template <> struct vnode_shift<1, 6, 33,  3> { static const int value =  92; };
template <> struct vnode_shift<1, 6, 33,  4> { static const int value =  0; };
// ILS = 6, CHECK_NODE = 34
template <> struct vnode_shift<1, 6, 34,  0> { static const int value =  4; };
template <> struct vnode_shift<1, 6, 34,  1> { static const int value =  153; };
template <> struct vnode_shift<1, 6, 34,  2> { static const int value =  197; };
template <> struct vnode_shift<1, 6, 34,  3> { static const int value =  155; };
template <> struct vnode_shift<1, 6, 34,  4> { static const int value =  0; };
// ILS = 6, CHECK_NODE = 35
template <> struct vnode_shift<1, 6, 35,  0> { static const int value =  4; };
template <> struct vnode_shift<1, 6, 35,  1> { static const int value =  45; };
template <> struct vnode_shift<1, 6, 35,  2> { static const int value =  168; };
template <> struct vnode_shift<1, 6, 35,  3> { static const int value =  185; };
template <> struct vnode_shift<1, 6, 35,  4> { static const int value =  0; };
// ILS = 6, CHECK_NODE = 36
template <> struct vnode_shift<1, 6, 36,  0> { static const int value =  6; };
template <> struct vnode_shift<1, 6, 36,  1> { static const int value =  200; };
template <> struct vnode_shift<1, 6, 36,  2> { static const int value =  177; };
template <> struct vnode_shift<1, 6, 36,  3> { static const int value =  43; };
template <> struct vnode_shift<1, 6, 36,  4> { static const int value =  0; };
// ILS = 6, CHECK_NODE = 37
template <> struct vnode_shift<1, 6, 37,  0> { static const int value =  82; };
template <> struct vnode_shift<1, 6, 37,  1> { static const int value =  2; };
template <> struct vnode_shift<1, 6, 37,  2> { static const int value =  135; };
template <> struct vnode_shift<1, 6, 37,  3> { static const int value =  0; };
// ILS = 6, CHECK_NODE = 38
template <> struct vnode_shift<1, 6, 38,  0> { static const int value =  91; };
template <> struct vnode_shift<1, 6, 38,  1> { static const int value =  64; };
template <> struct vnode_shift<1, 6, 38,  2> { static const int value =  198; };
template <> struct vnode_shift<1, 6, 38,  3> { static const int value =  100; };
template <> struct vnode_shift<1, 6, 38,  4> { static const int value =  0; };
// ILS = 6, CHECK_NODE = 39
template <> struct vnode_shift<1, 6, 39,  0> { static const int value =  4; };
template <> struct vnode_shift<1, 6, 39,  1> { static const int value =  28; };
template <> struct vnode_shift<1, 6, 39,  2> { static const int value =  109; };
template <> struct vnode_shift<1, 6, 39,  3> { static const int value =  188; };
template <> struct vnode_shift<1, 6, 39,  4> { static const int value =  0; };
// ILS = 6, CHECK_NODE = 40
template <> struct vnode_shift<1, 6, 40,  0> { static const int value =  10; };
template <> struct vnode_shift<1, 6, 40,  1> { static const int value =  84; };
template <> struct vnode_shift<1, 6, 40,  2> { static const int value =  12; };
template <> struct vnode_shift<1, 6, 40,  3> { static const int value =  0; };
// ILS = 6, CHECK_NODE = 41
template <> struct vnode_shift<1, 6, 41,  0> { static const int value =  2; };
template <> struct vnode_shift<1, 6, 41,  1> { static const int value =  75; };
template <> struct vnode_shift<1, 6, 41,  2> { static const int value =  142; };
template <> struct vnode_shift<1, 6, 41,  3> { static const int value =  128; };
template <> struct vnode_shift<1, 6, 41,  4> { static const int value =  0; };
// ILS = 6, CHECK_NODE = 42
template <> struct vnode_shift<1, 6, 42,  0> { static const int value =  163; };
template <> struct vnode_shift<1, 6, 42,  1> { static const int value =  10; };
template <> struct vnode_shift<1, 6, 42,  2> { static const int value =  162; };
template <> struct vnode_shift<1, 6, 42,  3> { static const int value =  0; };
// ILS = 6, CHECK_NODE = 43
template <> struct vnode_shift<1, 6, 43,  0> { static const int value =  1; };
template <> struct vnode_shift<1, 6, 43,  1> { static const int value =  163; };
template <> struct vnode_shift<1, 6, 43,  2> { static const int value =  99; };
template <> struct vnode_shift<1, 6, 43,  3> { static const int value =  98; };
template <> struct vnode_shift<1, 6, 43,  4> { static const int value =  0; };
// ILS = 6, CHECK_NODE = 44
template <> struct vnode_shift<1, 6, 44,  0> { static const int value =  4; };
template <> struct vnode_shift<1, 6, 44,  1> { static const int value =  6; };
template <> struct vnode_shift<1, 6, 44,  2> { static const int value =  142; };
template <> struct vnode_shift<1, 6, 44,  3> { static const int value =  3; };
template <> struct vnode_shift<1, 6, 44,  4> { static const int value =  0; };
// ILS = 6, CHECK_NODE = 45
template <> struct vnode_shift<1, 6, 45,  0> { static const int value =  181; };
template <> struct vnode_shift<1, 6, 45,  1> { static const int value =  45; };
template <> struct vnode_shift<1, 6, 45,  2> { static const int value =  153; };
template <> struct vnode_shift<1, 6, 45,  3> { static const int value =  0; };
// ILS = 7, CHECK_NODE = 0
template <> struct vnode_shift<1, 7, 0,  0> { static const int value =  135; };
template <> struct vnode_shift<1, 7, 0,  1> { static const int value =  227; };
template <> struct vnode_shift<1, 7, 0,  2> { static const int value =  126; };
template <> struct vnode_shift<1, 7, 0,  3> { static const int value =  134; };
template <> struct vnode_shift<1, 7, 0,  4> { static const int value =  84; };
template <> struct vnode_shift<1, 7, 0,  5> { static const int value =  83; };
template <> struct vnode_shift<1, 7, 0,  6> { static const int value =  53; };
template <> struct vnode_shift<1, 7, 0,  7> { static const int value =  225; };
template <> struct vnode_shift<1, 7, 0,  8> { static const int value =  205; };
template <> struct vnode_shift<1, 7, 0,  9> { static const int value =  128; };
template <> struct vnode_shift<1, 7, 0,  10> { static const int value =  75; };
template <> struct vnode_shift<1, 7, 0,  11> { static const int value =  135; };
template <> struct vnode_shift<1, 7, 0,  12> { static const int value =  217; };
template <> struct vnode_shift<1, 7, 0,  13> { static const int value =  220; };
template <> struct vnode_shift<1, 7, 0,  14> { static const int value =  90; };
template <> struct vnode_shift<1, 7, 0,  15> { static const int value =  105; };
template <> struct vnode_shift<1, 7, 0,  16> { static const int value =  137; };
template <> struct vnode_shift<1, 7, 0,  17> { static const int value =  1; };
template <> struct vnode_shift<1, 7, 0,  18> { static const int value =  0; };
// ILS = 7, CHECK_NODE = 1
template <> struct vnode_shift<1, 7, 1,  0> { static const int value =  96; };
template <> struct vnode_shift<1, 7, 1,  1> { static const int value =  236; };
template <> struct vnode_shift<1, 7, 1,  2> { static const int value =  136; };
template <> struct vnode_shift<1, 7, 1,  3> { static const int value =  221; };
template <> struct vnode_shift<1, 7, 1,  4> { static const int value =  128; };
template <> struct vnode_shift<1, 7, 1,  5> { static const int value =  92; };
template <> struct vnode_shift<1, 7, 1,  6> { static const int value =  172; };
template <> struct vnode_shift<1, 7, 1,  7> { static const int value =  56; };
template <> struct vnode_shift<1, 7, 1,  8> { static const int value =  11; };
template <> struct vnode_shift<1, 7, 1,  9> { static const int value =  189; };
template <> struct vnode_shift<1, 7, 1,  10> { static const int value =  95; };
template <> struct vnode_shift<1, 7, 1,  11> { static const int value =  85; };
template <> struct vnode_shift<1, 7, 1,  12> { static const int value =  153; };
template <> struct vnode_shift<1, 7, 1,  13> { static const int value =  87; };
template <> struct vnode_shift<1, 7, 1,  14> { static const int value =  163; };
template <> struct vnode_shift<1, 7, 1,  15> { static const int value =  216; };
template <> struct vnode_shift<1, 7, 1,  16> { static const int value =  0; };
template <> struct vnode_shift<1, 7, 1,  17> { static const int value =  0; };
template <> struct vnode_shift<1, 7, 1,  18> { static const int value =  0; };
// ILS = 7, CHECK_NODE = 2
template <> struct vnode_shift<1, 7, 2,  0> { static const int value =  189; };
template <> struct vnode_shift<1, 7, 2,  1> { static const int value =  4; };
template <> struct vnode_shift<1, 7, 2,  2> { static const int value =  225; };
template <> struct vnode_shift<1, 7, 2,  3> { static const int value =  151; };
template <> struct vnode_shift<1, 7, 2,  4> { static const int value =  236; };
template <> struct vnode_shift<1, 7, 2,  5> { static const int value =  117; };
template <> struct vnode_shift<1, 7, 2,  6> { static const int value =  179; };
template <> struct vnode_shift<1, 7, 2,  7> { static const int value =  92; };
template <> struct vnode_shift<1, 7, 2,  8> { static const int value =  24; };
template <> struct vnode_shift<1, 7, 2,  9> { static const int value =  68; };
template <> struct vnode_shift<1, 7, 2,  10> { static const int value =  6; };
template <> struct vnode_shift<1, 7, 2,  11> { static const int value =  101; };
template <> struct vnode_shift<1, 7, 2,  12> { static const int value =  33; };
template <> struct vnode_shift<1, 7, 2,  13> { static const int value =  96; };
template <> struct vnode_shift<1, 7, 2,  14> { static const int value =  125; };
template <> struct vnode_shift<1, 7, 2,  15> { static const int value =  67; };
template <> struct vnode_shift<1, 7, 2,  16> { static const int value =  230; };
template <> struct vnode_shift<1, 7, 2,  17> { static const int value =  0; };
template <> struct vnode_shift<1, 7, 2,  18> { static const int value =  0; };
// ILS = 7, CHECK_NODE = 3
template <> struct vnode_shift<1, 7, 3,  0> { static const int value =  128; };
template <> struct vnode_shift<1, 7, 3,  1> { static const int value =  23; };
template <> struct vnode_shift<1, 7, 3,  2> { static const int value =  162; };
template <> struct vnode_shift<1, 7, 3,  3> { static const int value =  220; };
template <> struct vnode_shift<1, 7, 3,  4> { static const int value =  43; };
template <> struct vnode_shift<1, 7, 3,  5> { static const int value =  186; };
template <> struct vnode_shift<1, 7, 3,  6> { static const int value =  96; };
template <> struct vnode_shift<1, 7, 3,  7> { static const int value =  1; };
template <> struct vnode_shift<1, 7, 3,  8> { static const int value =  216; };
template <> struct vnode_shift<1, 7, 3,  9> { static const int value =  22; };
template <> struct vnode_shift<1, 7, 3,  10> { static const int value =  24; };
template <> struct vnode_shift<1, 7, 3,  11> { static const int value =  167; };
template <> struct vnode_shift<1, 7, 3,  12> { static const int value =  200; };
template <> struct vnode_shift<1, 7, 3,  13> { static const int value =  32; };
template <> struct vnode_shift<1, 7, 3,  14> { static const int value =  235; };
template <> struct vnode_shift<1, 7, 3,  15> { static const int value =  172; };
template <> struct vnode_shift<1, 7, 3,  16> { static const int value =  219; };
template <> struct vnode_shift<1, 7, 3,  17> { static const int value =  1; };
template <> struct vnode_shift<1, 7, 3,  18> { static const int value =  0; };
// ILS = 7, CHECK_NODE = 4
template <> struct vnode_shift<1, 7, 4,  0> { static const int value =  64; };
template <> struct vnode_shift<1, 7, 4,  1> { static const int value =  211; };
template <> struct vnode_shift<1, 7, 4,  2> { static const int value =  0; };
// ILS = 7, CHECK_NODE = 5
template <> struct vnode_shift<1, 7, 5,  0> { static const int value =  2; };
template <> struct vnode_shift<1, 7, 5,  1> { static const int value =  171; };
template <> struct vnode_shift<1, 7, 5,  2> { static const int value =  47; };
template <> struct vnode_shift<1, 7, 5,  3> { static const int value =  143; };
template <> struct vnode_shift<1, 7, 5,  4> { static const int value =  210; };
template <> struct vnode_shift<1, 7, 5,  5> { static const int value =  180; };
template <> struct vnode_shift<1, 7, 5,  6> { static const int value =  180; };
template <> struct vnode_shift<1, 7, 5,  7> { static const int value =  0; };
// ILS = 7, CHECK_NODE = 6
template <> struct vnode_shift<1, 7, 6,  0> { static const int value =  199; };
template <> struct vnode_shift<1, 7, 6,  1> { static const int value =  22; };
template <> struct vnode_shift<1, 7, 6,  2> { static const int value =  23; };
template <> struct vnode_shift<1, 7, 6,  3> { static const int value =  100; };
template <> struct vnode_shift<1, 7, 6,  4> { static const int value =  92; };
template <> struct vnode_shift<1, 7, 6,  5> { static const int value =  207; };
template <> struct vnode_shift<1, 7, 6,  6> { static const int value =  52; };
template <> struct vnode_shift<1, 7, 6,  7> { static const int value =  13; };
template <> struct vnode_shift<1, 7, 6,  8> { static const int value =  0; };
// ILS = 7, CHECK_NODE = 7
template <> struct vnode_shift<1, 7, 7,  0> { static const int value =  77; };
template <> struct vnode_shift<1, 7, 7,  1> { static const int value =  146; };
template <> struct vnode_shift<1, 7, 7,  2> { static const int value =  209; };
template <> struct vnode_shift<1, 7, 7,  3> { static const int value =  32; };
template <> struct vnode_shift<1, 7, 7,  4> { static const int value =  166; };
template <> struct vnode_shift<1, 7, 7,  5> { static const int value =  18; };
template <> struct vnode_shift<1, 7, 7,  6> { static const int value =  0; };
// ILS = 7, CHECK_NODE = 8
template <> struct vnode_shift<1, 7, 8,  0> { static const int value =  181; };
template <> struct vnode_shift<1, 7, 8,  1> { static const int value =  105; };
template <> struct vnode_shift<1, 7, 8,  2> { static const int value =  141; };
template <> struct vnode_shift<1, 7, 8,  3> { static const int value =  223; };
template <> struct vnode_shift<1, 7, 8,  4> { static const int value =  177; };
template <> struct vnode_shift<1, 7, 8,  5> { static const int value =  145; };
template <> struct vnode_shift<1, 7, 8,  6> { static const int value =  199; };
template <> struct vnode_shift<1, 7, 8,  7> { static const int value =  153; };
template <> struct vnode_shift<1, 7, 8,  8> { static const int value =  38; };
template <> struct vnode_shift<1, 7, 8,  9> { static const int value =  0; };
// ILS = 7, CHECK_NODE = 9
template <> struct vnode_shift<1, 7, 9,  0> { static const int value =  169; };
template <> struct vnode_shift<1, 7, 9,  1> { static const int value =  12; };
template <> struct vnode_shift<1, 7, 9,  2> { static const int value =  206; };
template <> struct vnode_shift<1, 7, 9,  3> { static const int value =  221; };
template <> struct vnode_shift<1, 7, 9,  4> { static const int value =  17; };
template <> struct vnode_shift<1, 7, 9,  5> { static const int value =  212; };
template <> struct vnode_shift<1, 7, 9,  6> { static const int value =  92; };
template <> struct vnode_shift<1, 7, 9,  7> { static const int value =  205; };
template <> struct vnode_shift<1, 7, 9,  8> { static const int value =  0; };
// ILS = 7, CHECK_NODE = 10
template <> struct vnode_shift<1, 7, 10,  0> { static const int value =  116; };
template <> struct vnode_shift<1, 7, 10,  1> { static const int value =  151; };
template <> struct vnode_shift<1, 7, 10,  2> { static const int value =  70; };
template <> struct vnode_shift<1, 7, 10,  3> { static const int value =  230; };
template <> struct vnode_shift<1, 7, 10,  4> { static const int value =  115; };
template <> struct vnode_shift<1, 7, 10,  5> { static const int value =  84; };
template <> struct vnode_shift<1, 7, 10,  6> { static const int value =  0; };
// ILS = 7, CHECK_NODE = 11
template <> struct vnode_shift<1, 7, 11,  0> { static const int value =  45; };
template <> struct vnode_shift<1, 7, 11,  1> { static const int value =  115; };
template <> struct vnode_shift<1, 7, 11,  2> { static const int value =  134; };
template <> struct vnode_shift<1, 7, 11,  3> { static const int value =  1; };
template <> struct vnode_shift<1, 7, 11,  4> { static const int value =  152; };
template <> struct vnode_shift<1, 7, 11,  5> { static const int value =  165; };
template <> struct vnode_shift<1, 7, 11,  6> { static const int value =  107; };
template <> struct vnode_shift<1, 7, 11,  7> { static const int value =  0; };
// ILS = 7, CHECK_NODE = 12
template <> struct vnode_shift<1, 7, 12,  0> { static const int value =  186; };
template <> struct vnode_shift<1, 7, 12,  1> { static const int value =  215; };
template <> struct vnode_shift<1, 7, 12,  2> { static const int value =  124; };
template <> struct vnode_shift<1, 7, 12,  3> { static const int value =  180; };
template <> struct vnode_shift<1, 7, 12,  4> { static const int value =  98; };
template <> struct vnode_shift<1, 7, 12,  5> { static const int value =  80; };
template <> struct vnode_shift<1, 7, 12,  6> { static const int value =  0; };
// ILS = 7, CHECK_NODE = 13
template <> struct vnode_shift<1, 7, 13,  0> { static const int value =  220; };
template <> struct vnode_shift<1, 7, 13,  1> { static const int value =  185; };
template <> struct vnode_shift<1, 7, 13,  2> { static const int value =  154; };
template <> struct vnode_shift<1, 7, 13,  3> { static const int value =  178; };
template <> struct vnode_shift<1, 7, 13,  4> { static const int value =  150; };
template <> struct vnode_shift<1, 7, 13,  5> { static const int value =  0; };
// ILS = 7, CHECK_NODE = 14
template <> struct vnode_shift<1, 7, 14,  0> { static const int value =  124; };
template <> struct vnode_shift<1, 7, 14,  1> { static const int value =  144; };
template <> struct vnode_shift<1, 7, 14,  2> { static const int value =  182; };
template <> struct vnode_shift<1, 7, 14,  3> { static const int value =  95; };
template <> struct vnode_shift<1, 7, 14,  4> { static const int value =  72; };
template <> struct vnode_shift<1, 7, 14,  5> { static const int value =  76; };
template <> struct vnode_shift<1, 7, 14,  6> { static const int value =  0; };
// ILS = 7, CHECK_NODE = 15
template <> struct vnode_shift<1, 7, 15,  0> { static const int value =  39; };
template <> struct vnode_shift<1, 7, 15,  1> { static const int value =  138; };
template <> struct vnode_shift<1, 7, 15,  2> { static const int value =  220; };
template <> struct vnode_shift<1, 7, 15,  3> { static const int value =  173; };
template <> struct vnode_shift<1, 7, 15,  4> { static const int value =  142; };
template <> struct vnode_shift<1, 7, 15,  5> { static const int value =  49; };
template <> struct vnode_shift<1, 7, 15,  6> { static const int value =  0; };
// ILS = 7, CHECK_NODE = 16
template <> struct vnode_shift<1, 7, 16,  0> { static const int value =  78; };
template <> struct vnode_shift<1, 7, 16,  1> { static const int value =  152; };
template <> struct vnode_shift<1, 7, 16,  2> { static const int value =  84; };
template <> struct vnode_shift<1, 7, 16,  3> { static const int value =  5; };
template <> struct vnode_shift<1, 7, 16,  4> { static const int value =  205; };
template <> struct vnode_shift<1, 7, 16,  5> { static const int value =  0; };
// ILS = 7, CHECK_NODE = 17
template <> struct vnode_shift<1, 7, 17,  0> { static const int value =  183; };
template <> struct vnode_shift<1, 7, 17,  1> { static const int value =  112; };
template <> struct vnode_shift<1, 7, 17,  2> { static const int value =  106; };
template <> struct vnode_shift<1, 7, 17,  3> { static const int value =  219; };
template <> struct vnode_shift<1, 7, 17,  4> { static const int value =  129; };
template <> struct vnode_shift<1, 7, 17,  5> { static const int value =  0; };
// ILS = 7, CHECK_NODE = 18
template <> struct vnode_shift<1, 7, 18,  0> { static const int value =  183; };
template <> struct vnode_shift<1, 7, 18,  1> { static const int value =  215; };
template <> struct vnode_shift<1, 7, 18,  2> { static const int value =  180; };
template <> struct vnode_shift<1, 7, 18,  3> { static const int value =  143; };
template <> struct vnode_shift<1, 7, 18,  4> { static const int value =  14; };
template <> struct vnode_shift<1, 7, 18,  5> { static const int value =  0; };
// ILS = 7, CHECK_NODE = 19
template <> struct vnode_shift<1, 7, 19,  0> { static const int value =  179; };
template <> struct vnode_shift<1, 7, 19,  1> { static const int value =  108; };
template <> struct vnode_shift<1, 7, 19,  2> { static const int value =  159; };
template <> struct vnode_shift<1, 7, 19,  3> { static const int value =  138; };
template <> struct vnode_shift<1, 7, 19,  4> { static const int value =  196; };
template <> struct vnode_shift<1, 7, 19,  5> { static const int value =  0; };
// ILS = 7, CHECK_NODE = 20
template <> struct vnode_shift<1, 7, 20,  0> { static const int value =  77; };
template <> struct vnode_shift<1, 7, 20,  1> { static const int value =  187; };
template <> struct vnode_shift<1, 7, 20,  2> { static const int value =  203; };
template <> struct vnode_shift<1, 7, 20,  3> { static const int value =  167; };
template <> struct vnode_shift<1, 7, 20,  4> { static const int value =  130; };
template <> struct vnode_shift<1, 7, 20,  5> { static const int value =  0; };
// ILS = 7, CHECK_NODE = 21
template <> struct vnode_shift<1, 7, 21,  0> { static const int value =  197; };
template <> struct vnode_shift<1, 7, 21,  1> { static const int value =  122; };
template <> struct vnode_shift<1, 7, 21,  2> { static const int value =  215; };
template <> struct vnode_shift<1, 7, 21,  3> { static const int value =  65; };
template <> struct vnode_shift<1, 7, 21,  4> { static const int value =  216; };
template <> struct vnode_shift<1, 7, 21,  5> { static const int value =  0; };
// ILS = 7, CHECK_NODE = 22
template <> struct vnode_shift<1, 7, 22,  0> { static const int value =  25; };
template <> struct vnode_shift<1, 7, 22,  1> { static const int value =  47; };
template <> struct vnode_shift<1, 7, 22,  2> { static const int value =  126; };
template <> struct vnode_shift<1, 7, 22,  3> { static const int value =  178; };
template <> struct vnode_shift<1, 7, 22,  4> { static const int value =  0; };
// ILS = 7, CHECK_NODE = 23
template <> struct vnode_shift<1, 7, 23,  0> { static const int value =  185; };
template <> struct vnode_shift<1, 7, 23,  1> { static const int value =  127; };
template <> struct vnode_shift<1, 7, 23,  2> { static const int value =  117; };
template <> struct vnode_shift<1, 7, 23,  3> { static const int value =  199; };
template <> struct vnode_shift<1, 7, 23,  4> { static const int value =  0; };
// ILS = 7, CHECK_NODE = 24
template <> struct vnode_shift<1, 7, 24,  0> { static const int value =  32; };
template <> struct vnode_shift<1, 7, 24,  1> { static const int value =  178; };
template <> struct vnode_shift<1, 7, 24,  2> { static const int value =  2; };
template <> struct vnode_shift<1, 7, 24,  3> { static const int value =  156; };
template <> struct vnode_shift<1, 7, 24,  4> { static const int value =  58; };
template <> struct vnode_shift<1, 7, 24,  5> { static const int value =  0; };
// ILS = 7, CHECK_NODE = 25
template <> struct vnode_shift<1, 7, 25,  0> { static const int value =  27; };
template <> struct vnode_shift<1, 7, 25,  1> { static const int value =  141; };
template <> struct vnode_shift<1, 7, 25,  2> { static const int value =  11; };
template <> struct vnode_shift<1, 7, 25,  3> { static const int value =  181; };
template <> struct vnode_shift<1, 7, 25,  4> { static const int value =  0; };
// ILS = 7, CHECK_NODE = 26
template <> struct vnode_shift<1, 7, 26,  0> { static const int value =  163; };
template <> struct vnode_shift<1, 7, 26,  1> { static const int value =  131; };
template <> struct vnode_shift<1, 7, 26,  2> { static const int value =  169; };
template <> struct vnode_shift<1, 7, 26,  3> { static const int value =  98; };
template <> struct vnode_shift<1, 7, 26,  4> { static const int value =  0; };
// ILS = 7, CHECK_NODE = 27
template <> struct vnode_shift<1, 7, 27,  0> { static const int value =  165; };
template <> struct vnode_shift<1, 7, 27,  1> { static const int value =  232; };
template <> struct vnode_shift<1, 7, 27,  2> { static const int value =  9; };
template <> struct vnode_shift<1, 7, 27,  3> { static const int value =  0; };
// ILS = 7, CHECK_NODE = 28
template <> struct vnode_shift<1, 7, 28,  0> { static const int value =  32; };
template <> struct vnode_shift<1, 7, 28,  1> { static const int value =  43; };
template <> struct vnode_shift<1, 7, 28,  2> { static const int value =  200; };
template <> struct vnode_shift<1, 7, 28,  3> { static const int value =  205; };
template <> struct vnode_shift<1, 7, 28,  4> { static const int value =  0; };
// ILS = 7, CHECK_NODE = 29
template <> struct vnode_shift<1, 7, 29,  0> { static const int value =  232; };
template <> struct vnode_shift<1, 7, 29,  1> { static const int value =  32; };
template <> struct vnode_shift<1, 7, 29,  2> { static const int value =  118; };
template <> struct vnode_shift<1, 7, 29,  3> { static const int value =  103; };
template <> struct vnode_shift<1, 7, 29,  4> { static const int value =  0; };
// ILS = 7, CHECK_NODE = 30
template <> struct vnode_shift<1, 7, 30,  0> { static const int value =  170; };
template <> struct vnode_shift<1, 7, 30,  1> { static const int value =  199; };
template <> struct vnode_shift<1, 7, 30,  2> { static const int value =  26; };
template <> struct vnode_shift<1, 7, 30,  3> { static const int value =  105; };
template <> struct vnode_shift<1, 7, 30,  4> { static const int value =  0; };
// ILS = 7, CHECK_NODE = 31
template <> struct vnode_shift<1, 7, 31,  0> { static const int value =  73; };
template <> struct vnode_shift<1, 7, 31,  1> { static const int value =  149; };
template <> struct vnode_shift<1, 7, 31,  2> { static const int value =  175; };
template <> struct vnode_shift<1, 7, 31,  3> { static const int value =  108; };
template <> struct vnode_shift<1, 7, 31,  4> { static const int value =  0; };
// ILS = 7, CHECK_NODE = 32
template <> struct vnode_shift<1, 7, 32,  0> { static const int value =  103; };
template <> struct vnode_shift<1, 7, 32,  1> { static const int value =  110; };
template <> struct vnode_shift<1, 7, 32,  2> { static const int value =  151; };
template <> struct vnode_shift<1, 7, 32,  3> { static const int value =  211; };
template <> struct vnode_shift<1, 7, 32,  4> { static const int value =  0; };
// ILS = 7, CHECK_NODE = 33
template <> struct vnode_shift<1, 7, 33,  0> { static const int value =  199; };
template <> struct vnode_shift<1, 7, 33,  1> { static const int value =  132; };
template <> struct vnode_shift<1, 7, 33,  2> { static const int value =  172; };
template <> struct vnode_shift<1, 7, 33,  3> { static const int value =  65; };
template <> struct vnode_shift<1, 7, 33,  4> { static const int value =  0; };
// ILS = 7, CHECK_NODE = 34
template <> struct vnode_shift<1, 7, 34,  0> { static const int value =  161; };
template <> struct vnode_shift<1, 7, 34,  1> { static const int value =  237; };
template <> struct vnode_shift<1, 7, 34,  2> { static const int value =  142; };
template <> struct vnode_shift<1, 7, 34,  3> { static const int value =  180; };
template <> struct vnode_shift<1, 7, 34,  4> { static const int value =  0; };
// ILS = 7, CHECK_NODE = 35
template <> struct vnode_shift<1, 7, 35,  0> { static const int value =  231; };
template <> struct vnode_shift<1, 7, 35,  1> { static const int value =  174; };
template <> struct vnode_shift<1, 7, 35,  2> { static const int value =  145; };
template <> struct vnode_shift<1, 7, 35,  3> { static const int value =  100; };
template <> struct vnode_shift<1, 7, 35,  4> { static const int value =  0; };
// ILS = 7, CHECK_NODE = 36
template <> struct vnode_shift<1, 7, 36,  0> { static const int value =  11; };
template <> struct vnode_shift<1, 7, 36,  1> { static const int value =  207; };
template <> struct vnode_shift<1, 7, 36,  2> { static const int value =  42; };
template <> struct vnode_shift<1, 7, 36,  3> { static const int value =  100; };
template <> struct vnode_shift<1, 7, 36,  4> { static const int value =  0; };
// ILS = 7, CHECK_NODE = 37
template <> struct vnode_shift<1, 7, 37,  0> { static const int value =  59; };
template <> struct vnode_shift<1, 7, 37,  1> { static const int value =  204; };
template <> struct vnode_shift<1, 7, 37,  2> { static const int value =  161; };
template <> struct vnode_shift<1, 7, 37,  3> { static const int value =  0; };
// ILS = 7, CHECK_NODE = 38
template <> struct vnode_shift<1, 7, 38,  0> { static const int value =  121; };
template <> struct vnode_shift<1, 7, 38,  1> { static const int value =  90; };
template <> struct vnode_shift<1, 7, 38,  2> { static const int value =  26; };
template <> struct vnode_shift<1, 7, 38,  3> { static const int value =  140; };
template <> struct vnode_shift<1, 7, 38,  4> { static const int value =  0; };
// ILS = 7, CHECK_NODE = 39
template <> struct vnode_shift<1, 7, 39,  0> { static const int value =  115; };
template <> struct vnode_shift<1, 7, 39,  1> { static const int value =  188; };
template <> struct vnode_shift<1, 7, 39,  2> { static const int value =  168; };
template <> struct vnode_shift<1, 7, 39,  3> { static const int value =  52; };
template <> struct vnode_shift<1, 7, 39,  4> { static const int value =  0; };
// ILS = 7, CHECK_NODE = 40
template <> struct vnode_shift<1, 7, 40,  0> { static const int value =  4; };
template <> struct vnode_shift<1, 7, 40,  1> { static const int value =  103; };
template <> struct vnode_shift<1, 7, 40,  2> { static const int value =  30; };
template <> struct vnode_shift<1, 7, 40,  3> { static const int value =  0; };
// ILS = 7, CHECK_NODE = 41
template <> struct vnode_shift<1, 7, 41,  0> { static const int value =  53; };
template <> struct vnode_shift<1, 7, 41,  1> { static const int value =  189; };
template <> struct vnode_shift<1, 7, 41,  2> { static const int value =  215; };
template <> struct vnode_shift<1, 7, 41,  3> { static const int value =  24; };
template <> struct vnode_shift<1, 7, 41,  4> { static const int value =  0; };
// ILS = 7, CHECK_NODE = 42
template <> struct vnode_shift<1, 7, 42,  0> { static const int value =  222; };
template <> struct vnode_shift<1, 7, 42,  1> { static const int value =  170; };
template <> struct vnode_shift<1, 7, 42,  2> { static const int value =  71; };
template <> struct vnode_shift<1, 7, 42,  3> { static const int value =  0; };
// ILS = 7, CHECK_NODE = 43
template <> struct vnode_shift<1, 7, 43,  0> { static const int value =  22; };
template <> struct vnode_shift<1, 7, 43,  1> { static const int value =  127; };
template <> struct vnode_shift<1, 7, 43,  2> { static const int value =  49; };
template <> struct vnode_shift<1, 7, 43,  3> { static const int value =  125; };
template <> struct vnode_shift<1, 7, 43,  4> { static const int value =  0; };
// ILS = 7, CHECK_NODE = 44
template <> struct vnode_shift<1, 7, 44,  0> { static const int value =  191; };
template <> struct vnode_shift<1, 7, 44,  1> { static const int value =  211; };
template <> struct vnode_shift<1, 7, 44,  2> { static const int value =  187; };
template <> struct vnode_shift<1, 7, 44,  3> { static const int value =  148; };
template <> struct vnode_shift<1, 7, 44,  4> { static const int value =  0; };
// ILS = 7, CHECK_NODE = 45
template <> struct vnode_shift<1, 7, 45,  0> { static const int value =  177; };
template <> struct vnode_shift<1, 7, 45,  1> { static const int value =  114; };
template <> struct vnode_shift<1, 7, 45,  2> { static const int value =  93; };
template <> struct vnode_shift<1, 7, 45,  3> { static const int value =  0; };
// ILS = 0, CHECK_NODE = 0
template <> struct vnode_shift<2, 0, 0,  0> { static const int value =  9; };
template <> struct vnode_shift<2, 0, 0,  1> { static const int value =  117; };
template <> struct vnode_shift<2, 0, 0,  2> { static const int value =  204; };
template <> struct vnode_shift<2, 0, 0,  3> { static const int value =  26; };
template <> struct vnode_shift<2, 0, 0,  4> { static const int value =  189; };
template <> struct vnode_shift<2, 0, 0,  5> { static const int value =  205; };
template <> struct vnode_shift<2, 0, 0,  6> { static const int value =  0; };
template <> struct vnode_shift<2, 0, 0,  7> { static const int value =  0; };
// ILS = 0, CHECK_NODE = 1
template <> struct vnode_shift<2, 0, 1,  0> { static const int value =  167; };
template <> struct vnode_shift<2, 0, 1,  1> { static const int value =  166; };
template <> struct vnode_shift<2, 0, 1,  2> { static const int value =  253; };
template <> struct vnode_shift<2, 0, 1,  3> { static const int value =  125; };
template <> struct vnode_shift<2, 0, 1,  4> { static const int value =  226; };
template <> struct vnode_shift<2, 0, 1,  5> { static const int value =  156; };
template <> struct vnode_shift<2, 0, 1,  6> { static const int value =  224; };
template <> struct vnode_shift<2, 0, 1,  7> { static const int value =  252; };
template <> struct vnode_shift<2, 0, 1,  8> { static const int value =  0; };
template <> struct vnode_shift<2, 0, 1,  9> { static const int value =  0; };
// ILS = 0, CHECK_NODE = 2
template <> struct vnode_shift<2, 0, 2,  0> { static const int value =  81; };
template <> struct vnode_shift<2, 0, 2,  1> { static const int value =  114; };
template <> struct vnode_shift<2, 0, 2,  2> { static const int value =  44; };
template <> struct vnode_shift<2, 0, 2,  3> { static const int value =  52; };
template <> struct vnode_shift<2, 0, 2,  4> { static const int value =  240; };
template <> struct vnode_shift<2, 0, 2,  5> { static const int value =  1; };
template <> struct vnode_shift<2, 0, 2,  6> { static const int value =  0; };
template <> struct vnode_shift<2, 0, 2,  7> { static const int value =  0; };
// ILS = 0, CHECK_NODE = 3
template <> struct vnode_shift<2, 0, 3,  0> { static const int value =  8; };
template <> struct vnode_shift<2, 0, 3,  1> { static const int value =  58; };
template <> struct vnode_shift<2, 0, 3,  2> { static const int value =  158; };
template <> struct vnode_shift<2, 0, 3,  3> { static const int value =  104; };
template <> struct vnode_shift<2, 0, 3,  4> { static const int value =  209; };
template <> struct vnode_shift<2, 0, 3,  5> { static const int value =  54; };
template <> struct vnode_shift<2, 0, 3,  6> { static const int value =  18; };
template <> struct vnode_shift<2, 0, 3,  7> { static const int value =  128; };
template <> struct vnode_shift<2, 0, 3,  8> { static const int value =  0; };
template <> struct vnode_shift<2, 0, 3,  9> { static const int value =  0; };
// ILS = 0, CHECK_NODE = 4
template <> struct vnode_shift<2, 0, 4,  0> { static const int value =  179; };
template <> struct vnode_shift<2, 0, 4,  1> { static const int value =  214; };
template <> struct vnode_shift<2, 0, 4,  2> { static const int value =  71; };
template <> struct vnode_shift<2, 0, 4,  3> { static const int value =  0; };
// ILS = 0, CHECK_NODE = 5
template <> struct vnode_shift<2, 0, 5,  0> { static const int value =  231; };
template <> struct vnode_shift<2, 0, 5,  1> { static const int value =  41; };
template <> struct vnode_shift<2, 0, 5,  2> { static const int value =  194; };
template <> struct vnode_shift<2, 0, 5,  3> { static const int value =  159; };
template <> struct vnode_shift<2, 0, 5,  4> { static const int value =  103; };
template <> struct vnode_shift<2, 0, 5,  5> { static const int value =  0; };
// ILS = 0, CHECK_NODE = 6
template <> struct vnode_shift<2, 0, 6,  0> { static const int value =  155; };
template <> struct vnode_shift<2, 0, 6,  1> { static const int value =  228; };
template <> struct vnode_shift<2, 0, 6,  2> { static const int value =  45; };
template <> struct vnode_shift<2, 0, 6,  3> { static const int value =  28; };
template <> struct vnode_shift<2, 0, 6,  4> { static const int value =  158; };
template <> struct vnode_shift<2, 0, 6,  5> { static const int value =  0; };
// ILS = 0, CHECK_NODE = 7
template <> struct vnode_shift<2, 0, 7,  0> { static const int value =  129; };
template <> struct vnode_shift<2, 0, 7,  1> { static const int value =  147; };
template <> struct vnode_shift<2, 0, 7,  2> { static const int value =  140; };
template <> struct vnode_shift<2, 0, 7,  3> { static const int value =  3; };
template <> struct vnode_shift<2, 0, 7,  4> { static const int value =  116; };
template <> struct vnode_shift<2, 0, 7,  5> { static const int value =  0; };
// ILS = 0, CHECK_NODE = 8
template <> struct vnode_shift<2, 0, 8,  0> { static const int value =  142; };
template <> struct vnode_shift<2, 0, 8,  1> { static const int value =  94; };
template <> struct vnode_shift<2, 0, 8,  2> { static const int value =  230; };
template <> struct vnode_shift<2, 0, 8,  3> { static const int value =  0; };
// ILS = 0, CHECK_NODE = 9
template <> struct vnode_shift<2, 0, 9,  0> { static const int value =  203; };
template <> struct vnode_shift<2, 0, 9,  1> { static const int value =  205; };
template <> struct vnode_shift<2, 0, 9,  2> { static const int value =  61; };
template <> struct vnode_shift<2, 0, 9,  3> { static const int value =  247; };
template <> struct vnode_shift<2, 0, 9,  4> { static const int value =  0; };
// ILS = 0, CHECK_NODE = 10
template <> struct vnode_shift<2, 0, 10,  0> { static const int value =  11; };
template <> struct vnode_shift<2, 0, 10,  1> { static const int value =  185; };
template <> struct vnode_shift<2, 0, 10,  2> { static const int value =  0; };
template <> struct vnode_shift<2, 0, 10,  3> { static const int value =  117; };
template <> struct vnode_shift<2, 0, 10,  4> { static const int value =  0; };
// ILS = 0, CHECK_NODE = 11
template <> struct vnode_shift<2, 0, 11,  0> { static const int value =  11; };
template <> struct vnode_shift<2, 0, 11,  1> { static const int value =  236; };
template <> struct vnode_shift<2, 0, 11,  2> { static const int value =  210; };
template <> struct vnode_shift<2, 0, 11,  3> { static const int value =  56; };
template <> struct vnode_shift<2, 0, 11,  4> { static const int value =  0; };
// ILS = 0, CHECK_NODE = 12
template <> struct vnode_shift<2, 0, 12,  0> { static const int value =  63; };
template <> struct vnode_shift<2, 0, 12,  1> { static const int value =  111; };
template <> struct vnode_shift<2, 0, 12,  2> { static const int value =  14; };
template <> struct vnode_shift<2, 0, 12,  3> { static const int value =  0; };
// ILS = 0, CHECK_NODE = 13
template <> struct vnode_shift<2, 0, 13,  0> { static const int value =  83; };
template <> struct vnode_shift<2, 0, 13,  1> { static const int value =  2; };
template <> struct vnode_shift<2, 0, 13,  2> { static const int value =  38; };
template <> struct vnode_shift<2, 0, 13,  3> { static const int value =  222; };
template <> struct vnode_shift<2, 0, 13,  4> { static const int value =  0; };
// ILS = 0, CHECK_NODE = 14
template <> struct vnode_shift<2, 0, 14,  0> { static const int value =  115; };
template <> struct vnode_shift<2, 0, 14,  1> { static const int value =  145; };
template <> struct vnode_shift<2, 0, 14,  2> { static const int value =  3; };
template <> struct vnode_shift<2, 0, 14,  3> { static const int value =  232; };
template <> struct vnode_shift<2, 0, 14,  4> { static const int value =  0; };
// ILS = 0, CHECK_NODE = 15
template <> struct vnode_shift<2, 0, 15,  0> { static const int value =  51; };
template <> struct vnode_shift<2, 0, 15,  1> { static const int value =  175; };
template <> struct vnode_shift<2, 0, 15,  2> { static const int value =  213; };
template <> struct vnode_shift<2, 0, 15,  3> { static const int value =  0; };
// ILS = 0, CHECK_NODE = 16
template <> struct vnode_shift<2, 0, 16,  0> { static const int value =  203; };
template <> struct vnode_shift<2, 0, 16,  1> { static const int value =  142; };
template <> struct vnode_shift<2, 0, 16,  2> { static const int value =  8; };
template <> struct vnode_shift<2, 0, 16,  3> { static const int value =  242; };
template <> struct vnode_shift<2, 0, 16,  4> { static const int value =  0; };
// ILS = 0, CHECK_NODE = 17
template <> struct vnode_shift<2, 0, 17,  0> { static const int value =  254; };
template <> struct vnode_shift<2, 0, 17,  1> { static const int value =  124; };
template <> struct vnode_shift<2, 0, 17,  2> { static const int value =  114; };
template <> struct vnode_shift<2, 0, 17,  3> { static const int value =  64; };
template <> struct vnode_shift<2, 0, 17,  4> { static const int value =  0; };
// ILS = 0, CHECK_NODE = 18
template <> struct vnode_shift<2, 0, 18,  0> { static const int value =  220; };
template <> struct vnode_shift<2, 0, 18,  1> { static const int value =  194; };
template <> struct vnode_shift<2, 0, 18,  2> { static const int value =  50; };
template <> struct vnode_shift<2, 0, 18,  3> { static const int value =  0; };
// ILS = 0, CHECK_NODE = 19
template <> struct vnode_shift<2, 0, 19,  0> { static const int value =  87; };
template <> struct vnode_shift<2, 0, 19,  1> { static const int value =  20; };
template <> struct vnode_shift<2, 0, 19,  2> { static const int value =  185; };
template <> struct vnode_shift<2, 0, 19,  3> { static const int value =  0; };
// ILS = 0, CHECK_NODE = 20
template <> struct vnode_shift<2, 0, 20,  0> { static const int value =  26; };
template <> struct vnode_shift<2, 0, 20,  1> { static const int value =  105; };
template <> struct vnode_shift<2, 0, 20,  2> { static const int value =  29; };
template <> struct vnode_shift<2, 0, 20,  3> { static const int value =  0; };
// ILS = 0, CHECK_NODE = 21
template <> struct vnode_shift<2, 0, 21,  0> { static const int value =  76; };
template <> struct vnode_shift<2, 0, 21,  1> { static const int value =  42; };
template <> struct vnode_shift<2, 0, 21,  2> { static const int value =  210; };
template <> struct vnode_shift<2, 0, 21,  3> { static const int value =  0; };
// ILS = 0, CHECK_NODE = 22
template <> struct vnode_shift<2, 0, 22,  0> { static const int value =  222; };
template <> struct vnode_shift<2, 0, 22,  1> { static const int value =  63; };
template <> struct vnode_shift<2, 0, 22,  2> { static const int value =  0; };
// ILS = 0, CHECK_NODE = 23
template <> struct vnode_shift<2, 0, 23,  0> { static const int value =  23; };
template <> struct vnode_shift<2, 0, 23,  1> { static const int value =  235; };
template <> struct vnode_shift<2, 0, 23,  2> { static const int value =  238; };
template <> struct vnode_shift<2, 0, 23,  3> { static const int value =  0; };
// ILS = 0, CHECK_NODE = 24
template <> struct vnode_shift<2, 0, 24,  0> { static const int value =  46; };
template <> struct vnode_shift<2, 0, 24,  1> { static const int value =  139; };
template <> struct vnode_shift<2, 0, 24,  2> { static const int value =  8; };
template <> struct vnode_shift<2, 0, 24,  3> { static const int value =  0; };
// ILS = 0, CHECK_NODE = 25
template <> struct vnode_shift<2, 0, 25,  0> { static const int value =  228; };
template <> struct vnode_shift<2, 0, 25,  1> { static const int value =  156; };
template <> struct vnode_shift<2, 0, 25,  2> { static const int value =  0; };
// ILS = 0, CHECK_NODE = 26
template <> struct vnode_shift<2, 0, 26,  0> { static const int value =  29; };
template <> struct vnode_shift<2, 0, 26,  1> { static const int value =  143; };
template <> struct vnode_shift<2, 0, 26,  2> { static const int value =  160; };
template <> struct vnode_shift<2, 0, 26,  3> { static const int value =  122; };
template <> struct vnode_shift<2, 0, 26,  4> { static const int value =  0; };
// ILS = 0, CHECK_NODE = 27
template <> struct vnode_shift<2, 0, 27,  0> { static const int value =  8; };
template <> struct vnode_shift<2, 0, 27,  1> { static const int value =  151; };
template <> struct vnode_shift<2, 0, 27,  2> { static const int value =  0; };
// ILS = 0, CHECK_NODE = 28
template <> struct vnode_shift<2, 0, 28,  0> { static const int value =  98; };
template <> struct vnode_shift<2, 0, 28,  1> { static const int value =  101; };
template <> struct vnode_shift<2, 0, 28,  2> { static const int value =  135; };
template <> struct vnode_shift<2, 0, 28,  3> { static const int value =  0; };
// ILS = 0, CHECK_NODE = 29
template <> struct vnode_shift<2, 0, 29,  0> { static const int value =  18; };
template <> struct vnode_shift<2, 0, 29,  1> { static const int value =  28; };
template <> struct vnode_shift<2, 0, 29,  2> { static const int value =  0; };
// ILS = 0, CHECK_NODE = 30
template <> struct vnode_shift<2, 0, 30,  0> { static const int value =  71; };
template <> struct vnode_shift<2, 0, 30,  1> { static const int value =  240; };
template <> struct vnode_shift<2, 0, 30,  2> { static const int value =  9; };
template <> struct vnode_shift<2, 0, 30,  3> { static const int value =  84; };
template <> struct vnode_shift<2, 0, 30,  4> { static const int value =  0; };
// ILS = 0, CHECK_NODE = 31
template <> struct vnode_shift<2, 0, 31,  0> { static const int value =  106; };
template <> struct vnode_shift<2, 0, 31,  1> { static const int value =  1; };
template <> struct vnode_shift<2, 0, 31,  2> { static const int value =  0; };
// ILS = 0, CHECK_NODE = 32
template <> struct vnode_shift<2, 0, 32,  0> { static const int value =  242; };
template <> struct vnode_shift<2, 0, 32,  1> { static const int value =  44; };
template <> struct vnode_shift<2, 0, 32,  2> { static const int value =  166; };
template <> struct vnode_shift<2, 0, 32,  3> { static const int value =  0; };
// ILS = 0, CHECK_NODE = 33
template <> struct vnode_shift<2, 0, 33,  0> { static const int value =  132; };
template <> struct vnode_shift<2, 0, 33,  1> { static const int value =  164; };
template <> struct vnode_shift<2, 0, 33,  2> { static const int value =  235; };
template <> struct vnode_shift<2, 0, 33,  3> { static const int value =  0; };
// ILS = 0, CHECK_NODE = 34
template <> struct vnode_shift<2, 0, 34,  0> { static const int value =  147; };
template <> struct vnode_shift<2, 0, 34,  1> { static const int value =  85; };
template <> struct vnode_shift<2, 0, 34,  2> { static const int value =  36; };
template <> struct vnode_shift<2, 0, 34,  3> { static const int value =  0; };
// ILS = 0, CHECK_NODE = 35
template <> struct vnode_shift<2, 0, 35,  0> { static const int value =  57; };
template <> struct vnode_shift<2, 0, 35,  1> { static const int value =  40; };
template <> struct vnode_shift<2, 0, 35,  2> { static const int value =  63; };
template <> struct vnode_shift<2, 0, 35,  3> { static const int value =  0; };
// ILS = 0, CHECK_NODE = 36
template <> struct vnode_shift<2, 0, 36,  0> { static const int value =  140; };
template <> struct vnode_shift<2, 0, 36,  1> { static const int value =  38; };
template <> struct vnode_shift<2, 0, 36,  2> { static const int value =  154; };
template <> struct vnode_shift<2, 0, 36,  3> { static const int value =  0; };
// ILS = 0, CHECK_NODE = 37
template <> struct vnode_shift<2, 0, 37,  0> { static const int value =  219; };
template <> struct vnode_shift<2, 0, 37,  1> { static const int value =  151; };
template <> struct vnode_shift<2, 0, 37,  2> { static const int value =  0; };
// ILS = 0, CHECK_NODE = 38
template <> struct vnode_shift<2, 0, 38,  0> { static const int value =  31; };
template <> struct vnode_shift<2, 0, 38,  1> { static const int value =  66; };
template <> struct vnode_shift<2, 0, 38,  2> { static const int value =  38; };
template <> struct vnode_shift<2, 0, 38,  3> { static const int value =  0; };
// ILS = 0, CHECK_NODE = 39
template <> struct vnode_shift<2, 0, 39,  0> { static const int value =  239; };
template <> struct vnode_shift<2, 0, 39,  1> { static const int value =  172; };
template <> struct vnode_shift<2, 0, 39,  2> { static const int value =  34; };
template <> struct vnode_shift<2, 0, 39,  3> { static const int value =  0; };
// ILS = 0, CHECK_NODE = 40
template <> struct vnode_shift<2, 0, 40,  0> { static const int value =  0; };
template <> struct vnode_shift<2, 0, 40,  1> { static const int value =  75; };
template <> struct vnode_shift<2, 0, 40,  2> { static const int value =  120; };
template <> struct vnode_shift<2, 0, 40,  3> { static const int value =  0; };
// ILS = 0, CHECK_NODE = 41
template <> struct vnode_shift<2, 0, 41,  0> { static const int value =  129; };
template <> struct vnode_shift<2, 0, 41,  1> { static const int value =  229; };
template <> struct vnode_shift<2, 0, 41,  2> { static const int value =  118; };
template <> struct vnode_shift<2, 0, 41,  3> { static const int value =  0; };
// ILS = 1, CHECK_NODE = 0
template <> struct vnode_shift<2, 1, 0,  0> { static const int value =  174; };
template <> struct vnode_shift<2, 1, 0,  1> { static const int value =  97; };
template <> struct vnode_shift<2, 1, 0,  2> { static const int value =  166; };
template <> struct vnode_shift<2, 1, 0,  3> { static const int value =  66; };
template <> struct vnode_shift<2, 1, 0,  4> { static const int value =  71; };
template <> struct vnode_shift<2, 1, 0,  5> { static const int value =  172; };
template <> struct vnode_shift<2, 1, 0,  6> { static const int value =  0; };
template <> struct vnode_shift<2, 1, 0,  7> { static const int value =  0; };
// ILS = 1, CHECK_NODE = 1
template <> struct vnode_shift<2, 1, 1,  0> { static const int value =  27; };
template <> struct vnode_shift<2, 1, 1,  1> { static const int value =  36; };
template <> struct vnode_shift<2, 1, 1,  2> { static const int value =  48; };
template <> struct vnode_shift<2, 1, 1,  3> { static const int value =  92; };
template <> struct vnode_shift<2, 1, 1,  4> { static const int value =  31; };
template <> struct vnode_shift<2, 1, 1,  5> { static const int value =  187; };
template <> struct vnode_shift<2, 1, 1,  6> { static const int value =  185; };
template <> struct vnode_shift<2, 1, 1,  7> { static const int value =  3; };
template <> struct vnode_shift<2, 1, 1,  8> { static const int value =  0; };
template <> struct vnode_shift<2, 1, 1,  9> { static const int value =  0; };
// ILS = 1, CHECK_NODE = 2
template <> struct vnode_shift<2, 1, 2,  0> { static const int value =  25; };
template <> struct vnode_shift<2, 1, 2,  1> { static const int value =  114; };
template <> struct vnode_shift<2, 1, 2,  2> { static const int value =  117; };
template <> struct vnode_shift<2, 1, 2,  3> { static const int value =  110; };
template <> struct vnode_shift<2, 1, 2,  4> { static const int value =  114; };
template <> struct vnode_shift<2, 1, 2,  5> { static const int value =  1; };
template <> struct vnode_shift<2, 1, 2,  6> { static const int value =  0; };
template <> struct vnode_shift<2, 1, 2,  7> { static const int value =  0; };
// ILS = 1, CHECK_NODE = 3
template <> struct vnode_shift<2, 1, 3,  0> { static const int value =  136; };
template <> struct vnode_shift<2, 1, 3,  1> { static const int value =  175; };
template <> struct vnode_shift<2, 1, 3,  2> { static const int value =  113; };
template <> struct vnode_shift<2, 1, 3,  3> { static const int value =  72; };
template <> struct vnode_shift<2, 1, 3,  4> { static const int value =  123; };
template <> struct vnode_shift<2, 1, 3,  5> { static const int value =  118; };
template <> struct vnode_shift<2, 1, 3,  6> { static const int value =  28; };
template <> struct vnode_shift<2, 1, 3,  7> { static const int value =  186; };
template <> struct vnode_shift<2, 1, 3,  8> { static const int value =  0; };
template <> struct vnode_shift<2, 1, 3,  9> { static const int value =  0; };
// ILS = 1, CHECK_NODE = 4
template <> struct vnode_shift<2, 1, 4,  0> { static const int value =  72; };
template <> struct vnode_shift<2, 1, 4,  1> { static const int value =  74; };
template <> struct vnode_shift<2, 1, 4,  2> { static const int value =  29; };
template <> struct vnode_shift<2, 1, 4,  3> { static const int value =  0; };
// ILS = 1, CHECK_NODE = 5
template <> struct vnode_shift<2, 1, 5,  0> { static const int value =  10; };
template <> struct vnode_shift<2, 1, 5,  1> { static const int value =  44; };
template <> struct vnode_shift<2, 1, 5,  2> { static const int value =  121; };
template <> struct vnode_shift<2, 1, 5,  3> { static const int value =  80; };
template <> struct vnode_shift<2, 1, 5,  4> { static const int value =  48; };
template <> struct vnode_shift<2, 1, 5,  5> { static const int value =  0; };
// ILS = 1, CHECK_NODE = 6
template <> struct vnode_shift<2, 1, 6,  0> { static const int value =  129; };
template <> struct vnode_shift<2, 1, 6,  1> { static const int value =  92; };
template <> struct vnode_shift<2, 1, 6,  2> { static const int value =  100; };
template <> struct vnode_shift<2, 1, 6,  3> { static const int value =  49; };
template <> struct vnode_shift<2, 1, 6,  4> { static const int value =  184; };
template <> struct vnode_shift<2, 1, 6,  5> { static const int value =  0; };
// ILS = 1, CHECK_NODE = 7
template <> struct vnode_shift<2, 1, 7,  0> { static const int value =  80; };
template <> struct vnode_shift<2, 1, 7,  1> { static const int value =  186; };
template <> struct vnode_shift<2, 1, 7,  2> { static const int value =  16; };
template <> struct vnode_shift<2, 1, 7,  3> { static const int value =  102; };
template <> struct vnode_shift<2, 1, 7,  4> { static const int value =  143; };
template <> struct vnode_shift<2, 1, 7,  5> { static const int value =  0; };
// ILS = 1, CHECK_NODE = 8
template <> struct vnode_shift<2, 1, 8,  0> { static const int value =  118; };
template <> struct vnode_shift<2, 1, 8,  1> { static const int value =  70; };
template <> struct vnode_shift<2, 1, 8,  2> { static const int value =  152; };
template <> struct vnode_shift<2, 1, 8,  3> { static const int value =  0; };
// ILS = 1, CHECK_NODE = 9
template <> struct vnode_shift<2, 1, 9,  0> { static const int value =  28; };
template <> struct vnode_shift<2, 1, 9,  1> { static const int value =  132; };
template <> struct vnode_shift<2, 1, 9,  2> { static const int value =  185; };
template <> struct vnode_shift<2, 1, 9,  3> { static const int value =  178; };
template <> struct vnode_shift<2, 1, 9,  4> { static const int value =  0; };
// ILS = 1, CHECK_NODE = 10
template <> struct vnode_shift<2, 1, 10,  0> { static const int value =  59; };
template <> struct vnode_shift<2, 1, 10,  1> { static const int value =  104; };
template <> struct vnode_shift<2, 1, 10,  2> { static const int value =  22; };
template <> struct vnode_shift<2, 1, 10,  3> { static const int value =  52; };
template <> struct vnode_shift<2, 1, 10,  4> { static const int value =  0; };
// ILS = 1, CHECK_NODE = 11
template <> struct vnode_shift<2, 1, 11,  0> { static const int value =  32; };
template <> struct vnode_shift<2, 1, 11,  1> { static const int value =  92; };
template <> struct vnode_shift<2, 1, 11,  2> { static const int value =  174; };
template <> struct vnode_shift<2, 1, 11,  3> { static const int value =  154; };
template <> struct vnode_shift<2, 1, 11,  4> { static const int value =  0; };
// ILS = 1, CHECK_NODE = 12
template <> struct vnode_shift<2, 1, 12,  0> { static const int value =  39; };
template <> struct vnode_shift<2, 1, 12,  1> { static const int value =  93; };
template <> struct vnode_shift<2, 1, 12,  2> { static const int value =  11; };
template <> struct vnode_shift<2, 1, 12,  3> { static const int value =  0; };
// ILS = 1, CHECK_NODE = 13
template <> struct vnode_shift<2, 1, 13,  0> { static const int value =  49; };
template <> struct vnode_shift<2, 1, 13,  1> { static const int value =  125; };
template <> struct vnode_shift<2, 1, 13,  2> { static const int value =  35; };
template <> struct vnode_shift<2, 1, 13,  3> { static const int value =  166; };
template <> struct vnode_shift<2, 1, 13,  4> { static const int value =  0; };
// ILS = 1, CHECK_NODE = 14
template <> struct vnode_shift<2, 1, 14,  0> { static const int value =  19; };
template <> struct vnode_shift<2, 1, 14,  1> { static const int value =  118; };
template <> struct vnode_shift<2, 1, 14,  2> { static const int value =  21; };
template <> struct vnode_shift<2, 1, 14,  3> { static const int value =  163; };
template <> struct vnode_shift<2, 1, 14,  4> { static const int value =  0; };
// ILS = 1, CHECK_NODE = 15
template <> struct vnode_shift<2, 1, 15,  0> { static const int value =  68; };
template <> struct vnode_shift<2, 1, 15,  1> { static const int value =  63; };
template <> struct vnode_shift<2, 1, 15,  2> { static const int value =  81; };
template <> struct vnode_shift<2, 1, 15,  3> { static const int value =  0; };
// ILS = 1, CHECK_NODE = 16
template <> struct vnode_shift<2, 1, 16,  0> { static const int value =  87; };
template <> struct vnode_shift<2, 1, 16,  1> { static const int value =  177; };
template <> struct vnode_shift<2, 1, 16,  2> { static const int value =  135; };
template <> struct vnode_shift<2, 1, 16,  3> { static const int value =  64; };
template <> struct vnode_shift<2, 1, 16,  4> { static const int value =  0; };
// ILS = 1, CHECK_NODE = 17
template <> struct vnode_shift<2, 1, 17,  0> { static const int value =  158; };
template <> struct vnode_shift<2, 1, 17,  1> { static const int value =  23; };
template <> struct vnode_shift<2, 1, 17,  2> { static const int value =  9; };
template <> struct vnode_shift<2, 1, 17,  3> { static const int value =  6; };
template <> struct vnode_shift<2, 1, 17,  4> { static const int value =  0; };
// ILS = 1, CHECK_NODE = 18
template <> struct vnode_shift<2, 1, 18,  0> { static const int value =  186; };
template <> struct vnode_shift<2, 1, 18,  1> { static const int value =  6; };
template <> struct vnode_shift<2, 1, 18,  2> { static const int value =  46; };
template <> struct vnode_shift<2, 1, 18,  3> { static const int value =  0; };
// ILS = 1, CHECK_NODE = 19
template <> struct vnode_shift<2, 1, 19,  0> { static const int value =  58; };
template <> struct vnode_shift<2, 1, 19,  1> { static const int value =  42; };
template <> struct vnode_shift<2, 1, 19,  2> { static const int value =  156; };
template <> struct vnode_shift<2, 1, 19,  3> { static const int value =  0; };
// ILS = 1, CHECK_NODE = 20
template <> struct vnode_shift<2, 1, 20,  0> { static const int value =  76; };
template <> struct vnode_shift<2, 1, 20,  1> { static const int value =  61; };
template <> struct vnode_shift<2, 1, 20,  2> { static const int value =  153; };
template <> struct vnode_shift<2, 1, 20,  3> { static const int value =  0; };
// ILS = 1, CHECK_NODE = 21
template <> struct vnode_shift<2, 1, 21,  0> { static const int value =  157; };
template <> struct vnode_shift<2, 1, 21,  1> { static const int value =  175; };
template <> struct vnode_shift<2, 1, 21,  2> { static const int value =  67; };
template <> struct vnode_shift<2, 1, 21,  3> { static const int value =  0; };
// ILS = 1, CHECK_NODE = 22
template <> struct vnode_shift<2, 1, 22,  0> { static const int value =  20; };
template <> struct vnode_shift<2, 1, 22,  1> { static const int value =  52; };
template <> struct vnode_shift<2, 1, 22,  2> { static const int value =  0; };
// ILS = 1, CHECK_NODE = 23
template <> struct vnode_shift<2, 1, 23,  0> { static const int value =  106; };
template <> struct vnode_shift<2, 1, 23,  1> { static const int value =  86; };
template <> struct vnode_shift<2, 1, 23,  2> { static const int value =  95; };
template <> struct vnode_shift<2, 1, 23,  3> { static const int value =  0; };
// ILS = 1, CHECK_NODE = 24
template <> struct vnode_shift<2, 1, 24,  0> { static const int value =  182; };
template <> struct vnode_shift<2, 1, 24,  1> { static const int value =  153; };
template <> struct vnode_shift<2, 1, 24,  2> { static const int value =  64; };
template <> struct vnode_shift<2, 1, 24,  3> { static const int value =  0; };
// ILS = 1, CHECK_NODE = 25
template <> struct vnode_shift<2, 1, 25,  0> { static const int value =  45; };
template <> struct vnode_shift<2, 1, 25,  1> { static const int value =  21; };
template <> struct vnode_shift<2, 1, 25,  2> { static const int value =  0; };
// ILS = 1, CHECK_NODE = 26
template <> struct vnode_shift<2, 1, 26,  0> { static const int value =  67; };
template <> struct vnode_shift<2, 1, 26,  1> { static const int value =  137; };
template <> struct vnode_shift<2, 1, 26,  2> { static const int value =  55; };
template <> struct vnode_shift<2, 1, 26,  3> { static const int value =  85; };
template <> struct vnode_shift<2, 1, 26,  4> { static const int value =  0; };
// ILS = 1, CHECK_NODE = 27
template <> struct vnode_shift<2, 1, 27,  0> { static const int value =  103; };
template <> struct vnode_shift<2, 1, 27,  1> { static const int value =  50; };
template <> struct vnode_shift<2, 1, 27,  2> { static const int value =  0; };
// ILS = 1, CHECK_NODE = 28
template <> struct vnode_shift<2, 1, 28,  0> { static const int value =  70; };
template <> struct vnode_shift<2, 1, 28,  1> { static const int value =  111; };
template <> struct vnode_shift<2, 1, 28,  2> { static const int value =  168; };
template <> struct vnode_shift<2, 1, 28,  3> { static const int value =  0; };
// ILS = 1, CHECK_NODE = 29
template <> struct vnode_shift<2, 1, 29,  0> { static const int value =  110; };
template <> struct vnode_shift<2, 1, 29,  1> { static const int value =  17; };
template <> struct vnode_shift<2, 1, 29,  2> { static const int value =  0; };
// ILS = 1, CHECK_NODE = 30
template <> struct vnode_shift<2, 1, 30,  0> { static const int value =  120; };
template <> struct vnode_shift<2, 1, 30,  1> { static const int value =  154; };
template <> struct vnode_shift<2, 1, 30,  2> { static const int value =  52; };
template <> struct vnode_shift<2, 1, 30,  3> { static const int value =  56; };
template <> struct vnode_shift<2, 1, 30,  4> { static const int value =  0; };
// ILS = 1, CHECK_NODE = 31
template <> struct vnode_shift<2, 1, 31,  0> { static const int value =  3; };
template <> struct vnode_shift<2, 1, 31,  1> { static const int value =  170; };
template <> struct vnode_shift<2, 1, 31,  2> { static const int value =  0; };
// ILS = 1, CHECK_NODE = 32
template <> struct vnode_shift<2, 1, 32,  0> { static const int value =  84; };
template <> struct vnode_shift<2, 1, 32,  1> { static const int value =  8; };
template <> struct vnode_shift<2, 1, 32,  2> { static const int value =  17; };
template <> struct vnode_shift<2, 1, 32,  3> { static const int value =  0; };
// ILS = 1, CHECK_NODE = 33
template <> struct vnode_shift<2, 1, 33,  0> { static const int value =  165; };
template <> struct vnode_shift<2, 1, 33,  1> { static const int value =  179; };
template <> struct vnode_shift<2, 1, 33,  2> { static const int value =  124; };
template <> struct vnode_shift<2, 1, 33,  3> { static const int value =  0; };
// ILS = 1, CHECK_NODE = 34
template <> struct vnode_shift<2, 1, 34,  0> { static const int value =  173; };
template <> struct vnode_shift<2, 1, 34,  1> { static const int value =  177; };
template <> struct vnode_shift<2, 1, 34,  2> { static const int value =  12; };
template <> struct vnode_shift<2, 1, 34,  3> { static const int value =  0; };
// ILS = 1, CHECK_NODE = 35
template <> struct vnode_shift<2, 1, 35,  0> { static const int value =  77; };
template <> struct vnode_shift<2, 1, 35,  1> { static const int value =  184; };
template <> struct vnode_shift<2, 1, 35,  2> { static const int value =  18; };
template <> struct vnode_shift<2, 1, 35,  3> { static const int value =  0; };
// ILS = 1, CHECK_NODE = 36
template <> struct vnode_shift<2, 1, 36,  0> { static const int value =  25; };
template <> struct vnode_shift<2, 1, 36,  1> { static const int value =  151; };
template <> struct vnode_shift<2, 1, 36,  2> { static const int value =  170; };
template <> struct vnode_shift<2, 1, 36,  3> { static const int value =  0; };
// ILS = 1, CHECK_NODE = 37
template <> struct vnode_shift<2, 1, 37,  0> { static const int value =  37; };
template <> struct vnode_shift<2, 1, 37,  1> { static const int value =  31; };
template <> struct vnode_shift<2, 1, 37,  2> { static const int value =  0; };
// ILS = 1, CHECK_NODE = 38
template <> struct vnode_shift<2, 1, 38,  0> { static const int value =  84; };
template <> struct vnode_shift<2, 1, 38,  1> { static const int value =  151; };
template <> struct vnode_shift<2, 1, 38,  2> { static const int value =  190; };
template <> struct vnode_shift<2, 1, 38,  3> { static const int value =  0; };
// ILS = 1, CHECK_NODE = 39
template <> struct vnode_shift<2, 1, 39,  0> { static const int value =  93; };
template <> struct vnode_shift<2, 1, 39,  1> { static const int value =  132; };
template <> struct vnode_shift<2, 1, 39,  2> { static const int value =  57; };
template <> struct vnode_shift<2, 1, 39,  3> { static const int value =  0; };
// ILS = 1, CHECK_NODE = 40
template <> struct vnode_shift<2, 1, 40,  0> { static const int value =  103; };
template <> struct vnode_shift<2, 1, 40,  1> { static const int value =  107; };
template <> struct vnode_shift<2, 1, 40,  2> { static const int value =  163; };
template <> struct vnode_shift<2, 1, 40,  3> { static const int value =  0; };
// ILS = 1, CHECK_NODE = 41
template <> struct vnode_shift<2, 1, 41,  0> { static const int value =  147; };
template <> struct vnode_shift<2, 1, 41,  1> { static const int value =  7; };
template <> struct vnode_shift<2, 1, 41,  2> { static const int value =  60; };
template <> struct vnode_shift<2, 1, 41,  3> { static const int value =  0; };
// ILS = 2, CHECK_NODE = 0
template <> struct vnode_shift<2, 2, 0,  0> { static const int value =  0; };
template <> struct vnode_shift<2, 2, 0,  1> { static const int value =  0; };
template <> struct vnode_shift<2, 2, 0,  2> { static const int value =  0; };
template <> struct vnode_shift<2, 2, 0,  3> { static const int value =  0; };
template <> struct vnode_shift<2, 2, 0,  4> { static const int value =  0; };
template <> struct vnode_shift<2, 2, 0,  5> { static const int value =  0; };
template <> struct vnode_shift<2, 2, 0,  6> { static const int value =  0; };
template <> struct vnode_shift<2, 2, 0,  7> { static const int value =  0; };
// ILS = 2, CHECK_NODE = 1
template <> struct vnode_shift<2, 2, 1,  0> { static const int value =  137; };
template <> struct vnode_shift<2, 2, 1,  1> { static const int value =  124; };
template <> struct vnode_shift<2, 2, 1,  2> { static const int value =  0; };
template <> struct vnode_shift<2, 2, 1,  3> { static const int value =  0; };
template <> struct vnode_shift<2, 2, 1,  4> { static const int value =  88; };
template <> struct vnode_shift<2, 2, 1,  5> { static const int value =  0; };
template <> struct vnode_shift<2, 2, 1,  6> { static const int value =  0; };
template <> struct vnode_shift<2, 2, 1,  7> { static const int value =  55; };
template <> struct vnode_shift<2, 2, 1,  8> { static const int value =  0; };
template <> struct vnode_shift<2, 2, 1,  9> { static const int value =  0; };
// ILS = 2, CHECK_NODE = 2
template <> struct vnode_shift<2, 2, 2,  0> { static const int value =  20; };
template <> struct vnode_shift<2, 2, 2,  1> { static const int value =  94; };
template <> struct vnode_shift<2, 2, 2,  2> { static const int value =  99; };
template <> struct vnode_shift<2, 2, 2,  3> { static const int value =  9; };
template <> struct vnode_shift<2, 2, 2,  4> { static const int value =  108; };
template <> struct vnode_shift<2, 2, 2,  5> { static const int value =  1; };
template <> struct vnode_shift<2, 2, 2,  6> { static const int value =  0; };
template <> struct vnode_shift<2, 2, 2,  7> { static const int value =  0; };
// ILS = 2, CHECK_NODE = 3
template <> struct vnode_shift<2, 2, 3,  0> { static const int value =  38; };
template <> struct vnode_shift<2, 2, 3,  1> { static const int value =  15; };
template <> struct vnode_shift<2, 2, 3,  2> { static const int value =  102; };
template <> struct vnode_shift<2, 2, 3,  3> { static const int value =  146; };
template <> struct vnode_shift<2, 2, 3,  4> { static const int value =  12; };
template <> struct vnode_shift<2, 2, 3,  5> { static const int value =  57; };
template <> struct vnode_shift<2, 2, 3,  6> { static const int value =  53; };
template <> struct vnode_shift<2, 2, 3,  7> { static const int value =  46; };
template <> struct vnode_shift<2, 2, 3,  8> { static const int value =  0; };
template <> struct vnode_shift<2, 2, 3,  9> { static const int value =  0; };
// ILS = 2, CHECK_NODE = 4
template <> struct vnode_shift<2, 2, 4,  0> { static const int value =  0; };
template <> struct vnode_shift<2, 2, 4,  1> { static const int value =  136; };
template <> struct vnode_shift<2, 2, 4,  2> { static const int value =  157; };
template <> struct vnode_shift<2, 2, 4,  3> { static const int value =  0; };
// ILS = 2, CHECK_NODE = 5
template <> struct vnode_shift<2, 2, 5,  0> { static const int value =  0; };
template <> struct vnode_shift<2, 2, 5,  1> { static const int value =  131; };
template <> struct vnode_shift<2, 2, 5,  2> { static const int value =  142; };
template <> struct vnode_shift<2, 2, 5,  3> { static const int value =  141; };
template <> struct vnode_shift<2, 2, 5,  4> { static const int value =  64; };
template <> struct vnode_shift<2, 2, 5,  5> { static const int value =  0; };
// ILS = 2, CHECK_NODE = 6
template <> struct vnode_shift<2, 2, 6,  0> { static const int value =  0; };
template <> struct vnode_shift<2, 2, 6,  1> { static const int value =  124; };
template <> struct vnode_shift<2, 2, 6,  2> { static const int value =  99; };
template <> struct vnode_shift<2, 2, 6,  3> { static const int value =  45; };
template <> struct vnode_shift<2, 2, 6,  4> { static const int value =  148; };
template <> struct vnode_shift<2, 2, 6,  5> { static const int value =  0; };
// ILS = 2, CHECK_NODE = 7
template <> struct vnode_shift<2, 2, 7,  0> { static const int value =  0; };
template <> struct vnode_shift<2, 2, 7,  1> { static const int value =  45; };
template <> struct vnode_shift<2, 2, 7,  2> { static const int value =  148; };
template <> struct vnode_shift<2, 2, 7,  3> { static const int value =  96; };
template <> struct vnode_shift<2, 2, 7,  4> { static const int value =  78; };
template <> struct vnode_shift<2, 2, 7,  5> { static const int value =  0; };
// ILS = 2, CHECK_NODE = 8
template <> struct vnode_shift<2, 2, 8,  0> { static const int value =  0; };
template <> struct vnode_shift<2, 2, 8,  1> { static const int value =  65; };
template <> struct vnode_shift<2, 2, 8,  2> { static const int value =  87; };
template <> struct vnode_shift<2, 2, 8,  3> { static const int value =  0; };
// ILS = 2, CHECK_NODE = 9
template <> struct vnode_shift<2, 2, 9,  0> { static const int value =  0; };
template <> struct vnode_shift<2, 2, 9,  1> { static const int value =  97; };
template <> struct vnode_shift<2, 2, 9,  2> { static const int value =  51; };
template <> struct vnode_shift<2, 2, 9,  3> { static const int value =  85; };
template <> struct vnode_shift<2, 2, 9,  4> { static const int value =  0; };
// ILS = 2, CHECK_NODE = 10
template <> struct vnode_shift<2, 2, 10,  0> { static const int value =  0; };
template <> struct vnode_shift<2, 2, 10,  1> { static const int value =  17; };
template <> struct vnode_shift<2, 2, 10,  2> { static const int value =  156; };
template <> struct vnode_shift<2, 2, 10,  3> { static const int value =  20; };
template <> struct vnode_shift<2, 2, 10,  4> { static const int value =  0; };
// ILS = 2, CHECK_NODE = 11
template <> struct vnode_shift<2, 2, 11,  0> { static const int value =  0; };
template <> struct vnode_shift<2, 2, 11,  1> { static const int value =  7; };
template <> struct vnode_shift<2, 2, 11,  2> { static const int value =  4; };
template <> struct vnode_shift<2, 2, 11,  3> { static const int value =  2; };
template <> struct vnode_shift<2, 2, 11,  4> { static const int value =  0; };
// ILS = 2, CHECK_NODE = 12
template <> struct vnode_shift<2, 2, 12,  0> { static const int value =  0; };
template <> struct vnode_shift<2, 2, 12,  1> { static const int value =  113; };
template <> struct vnode_shift<2, 2, 12,  2> { static const int value =  48; };
template <> struct vnode_shift<2, 2, 12,  3> { static const int value =  0; };
// ILS = 2, CHECK_NODE = 13
template <> struct vnode_shift<2, 2, 13,  0> { static const int value =  0; };
template <> struct vnode_shift<2, 2, 13,  1> { static const int value =  112; };
template <> struct vnode_shift<2, 2, 13,  2> { static const int value =  102; };
template <> struct vnode_shift<2, 2, 13,  3> { static const int value =  26; };
template <> struct vnode_shift<2, 2, 13,  4> { static const int value =  0; };
// ILS = 2, CHECK_NODE = 14
template <> struct vnode_shift<2, 2, 14,  0> { static const int value =  0; };
template <> struct vnode_shift<2, 2, 14,  1> { static const int value =  138; };
template <> struct vnode_shift<2, 2, 14,  2> { static const int value =  57; };
template <> struct vnode_shift<2, 2, 14,  3> { static const int value =  27; };
template <> struct vnode_shift<2, 2, 14,  4> { static const int value =  0; };
// ILS = 2, CHECK_NODE = 15
template <> struct vnode_shift<2, 2, 15,  0> { static const int value =  0; };
template <> struct vnode_shift<2, 2, 15,  1> { static const int value =  73; };
template <> struct vnode_shift<2, 2, 15,  2> { static const int value =  99; };
template <> struct vnode_shift<2, 2, 15,  3> { static const int value =  0; };
// ILS = 2, CHECK_NODE = 16
template <> struct vnode_shift<2, 2, 16,  0> { static const int value =  0; };
template <> struct vnode_shift<2, 2, 16,  1> { static const int value =  79; };
template <> struct vnode_shift<2, 2, 16,  2> { static const int value =  111; };
template <> struct vnode_shift<2, 2, 16,  3> { static const int value =  143; };
template <> struct vnode_shift<2, 2, 16,  4> { static const int value =  0; };
// ILS = 2, CHECK_NODE = 17
template <> struct vnode_shift<2, 2, 17,  0> { static const int value =  0; };
template <> struct vnode_shift<2, 2, 17,  1> { static const int value =  24; };
template <> struct vnode_shift<2, 2, 17,  2> { static const int value =  109; };
template <> struct vnode_shift<2, 2, 17,  3> { static const int value =  18; };
template <> struct vnode_shift<2, 2, 17,  4> { static const int value =  0; };
// ILS = 2, CHECK_NODE = 18
template <> struct vnode_shift<2, 2, 18,  0> { static const int value =  0; };
template <> struct vnode_shift<2, 2, 18,  1> { static const int value =  18; };
template <> struct vnode_shift<2, 2, 18,  2> { static const int value =  86; };
template <> struct vnode_shift<2, 2, 18,  3> { static const int value =  0; };
// ILS = 2, CHECK_NODE = 19
template <> struct vnode_shift<2, 2, 19,  0> { static const int value =  0; };
template <> struct vnode_shift<2, 2, 19,  1> { static const int value =  158; };
template <> struct vnode_shift<2, 2, 19,  2> { static const int value =  154; };
template <> struct vnode_shift<2, 2, 19,  3> { static const int value =  0; };
// ILS = 2, CHECK_NODE = 20
template <> struct vnode_shift<2, 2, 20,  0> { static const int value =  0; };
template <> struct vnode_shift<2, 2, 20,  1> { static const int value =  148; };
template <> struct vnode_shift<2, 2, 20,  2> { static const int value =  104; };
template <> struct vnode_shift<2, 2, 20,  3> { static const int value =  0; };
// ILS = 2, CHECK_NODE = 21
template <> struct vnode_shift<2, 2, 21,  0> { static const int value =  0; };
template <> struct vnode_shift<2, 2, 21,  1> { static const int value =  17; };
template <> struct vnode_shift<2, 2, 21,  2> { static const int value =  33; };
template <> struct vnode_shift<2, 2, 21,  3> { static const int value =  0; };
// ILS = 2, CHECK_NODE = 22
template <> struct vnode_shift<2, 2, 22,  0> { static const int value =  0; };
template <> struct vnode_shift<2, 2, 22,  1> { static const int value =  4; };
template <> struct vnode_shift<2, 2, 22,  2> { static const int value =  0; };
// ILS = 2, CHECK_NODE = 23
template <> struct vnode_shift<2, 2, 23,  0> { static const int value =  0; };
template <> struct vnode_shift<2, 2, 23,  1> { static const int value =  75; };
template <> struct vnode_shift<2, 2, 23,  2> { static const int value =  158; };
template <> struct vnode_shift<2, 2, 23,  3> { static const int value =  0; };
// ILS = 2, CHECK_NODE = 24
template <> struct vnode_shift<2, 2, 24,  0> { static const int value =  0; };
template <> struct vnode_shift<2, 2, 24,  1> { static const int value =  69; };
template <> struct vnode_shift<2, 2, 24,  2> { static const int value =  87; };
template <> struct vnode_shift<2, 2, 24,  3> { static const int value =  0; };
// ILS = 2, CHECK_NODE = 25
template <> struct vnode_shift<2, 2, 25,  0> { static const int value =  0; };
template <> struct vnode_shift<2, 2, 25,  1> { static const int value =  65; };
template <> struct vnode_shift<2, 2, 25,  2> { static const int value =  0; };
// ILS = 2, CHECK_NODE = 26
template <> struct vnode_shift<2, 2, 26,  0> { static const int value =  0; };
template <> struct vnode_shift<2, 2, 26,  1> { static const int value =  100; };
template <> struct vnode_shift<2, 2, 26,  2> { static const int value =  13; };
template <> struct vnode_shift<2, 2, 26,  3> { static const int value =  7; };
template <> struct vnode_shift<2, 2, 26,  4> { static const int value =  0; };
// ILS = 2, CHECK_NODE = 27
template <> struct vnode_shift<2, 2, 27,  0> { static const int value =  0; };
template <> struct vnode_shift<2, 2, 27,  1> { static const int value =  32; };
template <> struct vnode_shift<2, 2, 27,  2> { static const int value =  0; };
// ILS = 2, CHECK_NODE = 28
template <> struct vnode_shift<2, 2, 28,  0> { static const int value =  0; };
template <> struct vnode_shift<2, 2, 28,  1> { static const int value =  126; };
template <> struct vnode_shift<2, 2, 28,  2> { static const int value =  110; };
template <> struct vnode_shift<2, 2, 28,  3> { static const int value =  0; };
// ILS = 2, CHECK_NODE = 29
template <> struct vnode_shift<2, 2, 29,  0> { static const int value =  0; };
template <> struct vnode_shift<2, 2, 29,  1> { static const int value =  154; };
template <> struct vnode_shift<2, 2, 29,  2> { static const int value =  0; };
// ILS = 2, CHECK_NODE = 30
template <> struct vnode_shift<2, 2, 30,  0> { static const int value =  0; };
template <> struct vnode_shift<2, 2, 30,  1> { static const int value =  35; };
template <> struct vnode_shift<2, 2, 30,  2> { static const int value =  51; };
template <> struct vnode_shift<2, 2, 30,  3> { static const int value =  134; };
template <> struct vnode_shift<2, 2, 30,  4> { static const int value =  0; };
// ILS = 2, CHECK_NODE = 31
template <> struct vnode_shift<2, 2, 31,  0> { static const int value =  0; };
template <> struct vnode_shift<2, 2, 31,  1> { static const int value =  20; };
template <> struct vnode_shift<2, 2, 31,  2> { static const int value =  0; };
// ILS = 2, CHECK_NODE = 32
template <> struct vnode_shift<2, 2, 32,  0> { static const int value =  0; };
template <> struct vnode_shift<2, 2, 32,  1> { static const int value =  20; };
template <> struct vnode_shift<2, 2, 32,  2> { static const int value =  122; };
template <> struct vnode_shift<2, 2, 32,  3> { static const int value =  0; };
// ILS = 2, CHECK_NODE = 33
template <> struct vnode_shift<2, 2, 33,  0> { static const int value =  0; };
template <> struct vnode_shift<2, 2, 33,  1> { static const int value =  88; };
template <> struct vnode_shift<2, 2, 33,  2> { static const int value =  13; };
template <> struct vnode_shift<2, 2, 33,  3> { static const int value =  0; };
// ILS = 2, CHECK_NODE = 34
template <> struct vnode_shift<2, 2, 34,  0> { static const int value =  0; };
template <> struct vnode_shift<2, 2, 34,  1> { static const int value =  19; };
template <> struct vnode_shift<2, 2, 34,  2> { static const int value =  78; };
template <> struct vnode_shift<2, 2, 34,  3> { static const int value =  0; };
// ILS = 2, CHECK_NODE = 35
template <> struct vnode_shift<2, 2, 35,  0> { static const int value =  0; };
template <> struct vnode_shift<2, 2, 35,  1> { static const int value =  157; };
template <> struct vnode_shift<2, 2, 35,  2> { static const int value =  6; };
template <> struct vnode_shift<2, 2, 35,  3> { static const int value =  0; };
// ILS = 2, CHECK_NODE = 36
template <> struct vnode_shift<2, 2, 36,  0> { static const int value =  0; };
template <> struct vnode_shift<2, 2, 36,  1> { static const int value =  63; };
template <> struct vnode_shift<2, 2, 36,  2> { static const int value =  82; };
template <> struct vnode_shift<2, 2, 36,  3> { static const int value =  0; };
// ILS = 2, CHECK_NODE = 37
template <> struct vnode_shift<2, 2, 37,  0> { static const int value =  0; };
template <> struct vnode_shift<2, 2, 37,  1> { static const int value =  144; };
template <> struct vnode_shift<2, 2, 37,  2> { static const int value =  0; };
// ILS = 2, CHECK_NODE = 38
template <> struct vnode_shift<2, 2, 38,  0> { static const int value =  0; };
template <> struct vnode_shift<2, 2, 38,  1> { static const int value =  93; };
template <> struct vnode_shift<2, 2, 38,  2> { static const int value =  19; };
template <> struct vnode_shift<2, 2, 38,  3> { static const int value =  0; };
// ILS = 2, CHECK_NODE = 39
template <> struct vnode_shift<2, 2, 39,  0> { static const int value =  0; };
template <> struct vnode_shift<2, 2, 39,  1> { static const int value =  24; };
template <> struct vnode_shift<2, 2, 39,  2> { static const int value =  138; };
template <> struct vnode_shift<2, 2, 39,  3> { static const int value =  0; };
// ILS = 2, CHECK_NODE = 40
template <> struct vnode_shift<2, 2, 40,  0> { static const int value =  0; };
template <> struct vnode_shift<2, 2, 40,  1> { static const int value =  36; };
template <> struct vnode_shift<2, 2, 40,  2> { static const int value =  143; };
template <> struct vnode_shift<2, 2, 40,  3> { static const int value =  0; };
// ILS = 2, CHECK_NODE = 41
template <> struct vnode_shift<2, 2, 41,  0> { static const int value =  0; };
template <> struct vnode_shift<2, 2, 41,  1> { static const int value =  2; };
template <> struct vnode_shift<2, 2, 41,  2> { static const int value =  55; };
template <> struct vnode_shift<2, 2, 41,  3> { static const int value =  0; };
// ILS = 3, CHECK_NODE = 0
template <> struct vnode_shift<2, 3, 0,  0> { static const int value =  72; };
template <> struct vnode_shift<2, 3, 0,  1> { static const int value =  110; };
template <> struct vnode_shift<2, 3, 0,  2> { static const int value =  23; };
template <> struct vnode_shift<2, 3, 0,  3> { static const int value =  181; };
template <> struct vnode_shift<2, 3, 0,  4> { static const int value =  95; };
template <> struct vnode_shift<2, 3, 0,  5> { static const int value =  8; };
template <> struct vnode_shift<2, 3, 0,  6> { static const int value =  1; };
template <> struct vnode_shift<2, 3, 0,  7> { static const int value =  0; };
// ILS = 3, CHECK_NODE = 1
template <> struct vnode_shift<2, 3, 1,  0> { static const int value =  53; };
template <> struct vnode_shift<2, 3, 1,  1> { static const int value =  156; };
template <> struct vnode_shift<2, 3, 1,  2> { static const int value =  115; };
template <> struct vnode_shift<2, 3, 1,  3> { static const int value =  156; };
template <> struct vnode_shift<2, 3, 1,  4> { static const int value =  115; };
template <> struct vnode_shift<2, 3, 1,  5> { static const int value =  200; };
template <> struct vnode_shift<2, 3, 1,  6> { static const int value =  29; };
template <> struct vnode_shift<2, 3, 1,  7> { static const int value =  31; };
template <> struct vnode_shift<2, 3, 1,  8> { static const int value =  0; };
template <> struct vnode_shift<2, 3, 1,  9> { static const int value =  0; };
// ILS = 3, CHECK_NODE = 2
template <> struct vnode_shift<2, 3, 2,  0> { static const int value =  152; };
template <> struct vnode_shift<2, 3, 2,  1> { static const int value =  131; };
template <> struct vnode_shift<2, 3, 2,  2> { static const int value =  46; };
template <> struct vnode_shift<2, 3, 2,  3> { static const int value =  191; };
template <> struct vnode_shift<2, 3, 2,  4> { static const int value =  91; };
template <> struct vnode_shift<2, 3, 2,  5> { static const int value =  0; };
template <> struct vnode_shift<2, 3, 2,  6> { static const int value =  0; };
template <> struct vnode_shift<2, 3, 2,  7> { static const int value =  0; };
// ILS = 3, CHECK_NODE = 3
template <> struct vnode_shift<2, 3, 3,  0> { static const int value =  185; };
template <> struct vnode_shift<2, 3, 3,  1> { static const int value =  6; };
template <> struct vnode_shift<2, 3, 3,  2> { static const int value =  36; };
template <> struct vnode_shift<2, 3, 3,  3> { static const int value =  124; };
template <> struct vnode_shift<2, 3, 3,  4> { static const int value =  124; };
template <> struct vnode_shift<2, 3, 3,  5> { static const int value =  110; };
template <> struct vnode_shift<2, 3, 3,  6> { static const int value =  156; };
template <> struct vnode_shift<2, 3, 3,  7> { static const int value =  133; };
template <> struct vnode_shift<2, 3, 3,  8> { static const int value =  1; };
template <> struct vnode_shift<2, 3, 3,  9> { static const int value =  0; };
// ILS = 3, CHECK_NODE = 4
template <> struct vnode_shift<2, 3, 4,  0> { static const int value =  200; };
template <> struct vnode_shift<2, 3, 4,  1> { static const int value =  16; };
template <> struct vnode_shift<2, 3, 4,  2> { static const int value =  101; };
template <> struct vnode_shift<2, 3, 4,  3> { static const int value =  0; };
// ILS = 3, CHECK_NODE = 5
template <> struct vnode_shift<2, 3, 5,  0> { static const int value =  185; };
template <> struct vnode_shift<2, 3, 5,  1> { static const int value =  138; };
template <> struct vnode_shift<2, 3, 5,  2> { static const int value =  170; };
template <> struct vnode_shift<2, 3, 5,  3> { static const int value =  219; };
template <> struct vnode_shift<2, 3, 5,  4> { static const int value =  193; };
template <> struct vnode_shift<2, 3, 5,  5> { static const int value =  0; };
// ILS = 3, CHECK_NODE = 6
template <> struct vnode_shift<2, 3, 6,  0> { static const int value =  123; };
template <> struct vnode_shift<2, 3, 6,  1> { static const int value =  55; };
template <> struct vnode_shift<2, 3, 6,  2> { static const int value =  31; };
template <> struct vnode_shift<2, 3, 6,  3> { static const int value =  222; };
template <> struct vnode_shift<2, 3, 6,  4> { static const int value =  209; };
template <> struct vnode_shift<2, 3, 6,  5> { static const int value =  0; };
// ILS = 3, CHECK_NODE = 7
template <> struct vnode_shift<2, 3, 7,  0> { static const int value =  103; };
template <> struct vnode_shift<2, 3, 7,  1> { static const int value =  13; };
template <> struct vnode_shift<2, 3, 7,  2> { static const int value =  105; };
template <> struct vnode_shift<2, 3, 7,  3> { static const int value =  150; };
template <> struct vnode_shift<2, 3, 7,  4> { static const int value =  181; };
template <> struct vnode_shift<2, 3, 7,  5> { static const int value =  0; };
// ILS = 3, CHECK_NODE = 8
template <> struct vnode_shift<2, 3, 8,  0> { static const int value =  147; };
template <> struct vnode_shift<2, 3, 8,  1> { static const int value =  43; };
template <> struct vnode_shift<2, 3, 8,  2> { static const int value =  152; };
template <> struct vnode_shift<2, 3, 8,  3> { static const int value =  0; };
// ILS = 3, CHECK_NODE = 9
template <> struct vnode_shift<2, 3, 9,  0> { static const int value =  2; };
template <> struct vnode_shift<2, 3, 9,  1> { static const int value =  30; };
template <> struct vnode_shift<2, 3, 9,  2> { static const int value =  184; };
template <> struct vnode_shift<2, 3, 9,  3> { static const int value =  83; };
template <> struct vnode_shift<2, 3, 9,  4> { static const int value =  0; };
// ILS = 3, CHECK_NODE = 10
template <> struct vnode_shift<2, 3, 10,  0> { static const int value =  174; };
template <> struct vnode_shift<2, 3, 10,  1> { static const int value =  150; };
template <> struct vnode_shift<2, 3, 10,  2> { static const int value =  8; };
template <> struct vnode_shift<2, 3, 10,  3> { static const int value =  56; };
template <> struct vnode_shift<2, 3, 10,  4> { static const int value =  0; };
// ILS = 3, CHECK_NODE = 11
template <> struct vnode_shift<2, 3, 11,  0> { static const int value =  99; };
template <> struct vnode_shift<2, 3, 11,  1> { static const int value =  138; };
template <> struct vnode_shift<2, 3, 11,  2> { static const int value =  110; };
template <> struct vnode_shift<2, 3, 11,  3> { static const int value =  99; };
template <> struct vnode_shift<2, 3, 11,  4> { static const int value =  0; };
// ILS = 3, CHECK_NODE = 12
template <> struct vnode_shift<2, 3, 12,  0> { static const int value =  46; };
template <> struct vnode_shift<2, 3, 12,  1> { static const int value =  217; };
template <> struct vnode_shift<2, 3, 12,  2> { static const int value =  109; };
template <> struct vnode_shift<2, 3, 12,  3> { static const int value =  0; };
// ILS = 3, CHECK_NODE = 13
template <> struct vnode_shift<2, 3, 13,  0> { static const int value =  37; };
template <> struct vnode_shift<2, 3, 13,  1> { static const int value =  113; };
template <> struct vnode_shift<2, 3, 13,  2> { static const int value =  143; };
template <> struct vnode_shift<2, 3, 13,  3> { static const int value =  140; };
template <> struct vnode_shift<2, 3, 13,  4> { static const int value =  0; };
// ILS = 3, CHECK_NODE = 14
template <> struct vnode_shift<2, 3, 14,  0> { static const int value =  36; };
template <> struct vnode_shift<2, 3, 14,  1> { static const int value =  95; };
template <> struct vnode_shift<2, 3, 14,  2> { static const int value =  40; };
template <> struct vnode_shift<2, 3, 14,  3> { static const int value =  116; };
template <> struct vnode_shift<2, 3, 14,  4> { static const int value =  0; };
// ILS = 3, CHECK_NODE = 15
template <> struct vnode_shift<2, 3, 15,  0> { static const int value =  116; };
template <> struct vnode_shift<2, 3, 15,  1> { static const int value =  200; };
template <> struct vnode_shift<2, 3, 15,  2> { static const int value =  110; };
template <> struct vnode_shift<2, 3, 15,  3> { static const int value =  0; };
// ILS = 3, CHECK_NODE = 16
template <> struct vnode_shift<2, 3, 16,  0> { static const int value =  75; };
template <> struct vnode_shift<2, 3, 16,  1> { static const int value =  158; };
template <> struct vnode_shift<2, 3, 16,  2> { static const int value =  134; };
template <> struct vnode_shift<2, 3, 16,  3> { static const int value =  97; };
template <> struct vnode_shift<2, 3, 16,  4> { static const int value =  0; };
// ILS = 3, CHECK_NODE = 17
template <> struct vnode_shift<2, 3, 17,  0> { static const int value =  48; };
template <> struct vnode_shift<2, 3, 17,  1> { static const int value =  132; };
template <> struct vnode_shift<2, 3, 17,  2> { static const int value =  206; };
template <> struct vnode_shift<2, 3, 17,  3> { static const int value =  2; };
template <> struct vnode_shift<2, 3, 17,  4> { static const int value =  0; };
// ILS = 3, CHECK_NODE = 18
template <> struct vnode_shift<2, 3, 18,  0> { static const int value =  68; };
template <> struct vnode_shift<2, 3, 18,  1> { static const int value =  16; };
template <> struct vnode_shift<2, 3, 18,  2> { static const int value =  156; };
template <> struct vnode_shift<2, 3, 18,  3> { static const int value =  0; };
// ILS = 3, CHECK_NODE = 19
template <> struct vnode_shift<2, 3, 19,  0> { static const int value =  35; };
template <> struct vnode_shift<2, 3, 19,  1> { static const int value =  138; };
template <> struct vnode_shift<2, 3, 19,  2> { static const int value =  86; };
template <> struct vnode_shift<2, 3, 19,  3> { static const int value =  0; };
// ILS = 3, CHECK_NODE = 20
template <> struct vnode_shift<2, 3, 20,  0> { static const int value =  6; };
template <> struct vnode_shift<2, 3, 20,  1> { static const int value =  20; };
template <> struct vnode_shift<2, 3, 20,  2> { static const int value =  141; };
template <> struct vnode_shift<2, 3, 20,  3> { static const int value =  0; };
// ILS = 3, CHECK_NODE = 21
template <> struct vnode_shift<2, 3, 21,  0> { static const int value =  80; };
template <> struct vnode_shift<2, 3, 21,  1> { static const int value =  43; };
template <> struct vnode_shift<2, 3, 21,  2> { static const int value =  81; };
template <> struct vnode_shift<2, 3, 21,  3> { static const int value =  0; };
// ILS = 3, CHECK_NODE = 22
template <> struct vnode_shift<2, 3, 22,  0> { static const int value =  49; };
template <> struct vnode_shift<2, 3, 22,  1> { static const int value =  1; };
template <> struct vnode_shift<2, 3, 22,  2> { static const int value =  0; };
// ILS = 3, CHECK_NODE = 23
template <> struct vnode_shift<2, 3, 23,  0> { static const int value =  156; };
template <> struct vnode_shift<2, 3, 23,  1> { static const int value =  54; };
template <> struct vnode_shift<2, 3, 23,  2> { static const int value =  134; };
template <> struct vnode_shift<2, 3, 23,  3> { static const int value =  0; };
// ILS = 3, CHECK_NODE = 24
template <> struct vnode_shift<2, 3, 24,  0> { static const int value =  153; };
template <> struct vnode_shift<2, 3, 24,  1> { static const int value =  88; };
template <> struct vnode_shift<2, 3, 24,  2> { static const int value =  63; };
template <> struct vnode_shift<2, 3, 24,  3> { static const int value =  0; };
// ILS = 3, CHECK_NODE = 25
template <> struct vnode_shift<2, 3, 25,  0> { static const int value =  211; };
template <> struct vnode_shift<2, 3, 25,  1> { static const int value =  94; };
template <> struct vnode_shift<2, 3, 25,  2> { static const int value =  0; };
// ILS = 3, CHECK_NODE = 26
template <> struct vnode_shift<2, 3, 26,  0> { static const int value =  90; };
template <> struct vnode_shift<2, 3, 26,  1> { static const int value =  6; };
template <> struct vnode_shift<2, 3, 26,  2> { static const int value =  221; };
template <> struct vnode_shift<2, 3, 26,  3> { static const int value =  6; };
template <> struct vnode_shift<2, 3, 26,  4> { static const int value =  0; };
// ILS = 3, CHECK_NODE = 27
template <> struct vnode_shift<2, 3, 27,  0> { static const int value =  27; };
template <> struct vnode_shift<2, 3, 27,  1> { static const int value =  118; };
template <> struct vnode_shift<2, 3, 27,  2> { static const int value =  0; };
// ILS = 3, CHECK_NODE = 28
template <> struct vnode_shift<2, 3, 28,  0> { static const int value =  216; };
template <> struct vnode_shift<2, 3, 28,  1> { static const int value =  212; };
template <> struct vnode_shift<2, 3, 28,  2> { static const int value =  193; };
template <> struct vnode_shift<2, 3, 28,  3> { static const int value =  0; };
// ILS = 3, CHECK_NODE = 29
template <> struct vnode_shift<2, 3, 29,  0> { static const int value =  108; };
template <> struct vnode_shift<2, 3, 29,  1> { static const int value =  61; };
template <> struct vnode_shift<2, 3, 29,  2> { static const int value =  0; };
// ILS = 3, CHECK_NODE = 30
template <> struct vnode_shift<2, 3, 30,  0> { static const int value =  106; };
template <> struct vnode_shift<2, 3, 30,  1> { static const int value =  44; };
template <> struct vnode_shift<2, 3, 30,  2> { static const int value =  185; };
template <> struct vnode_shift<2, 3, 30,  3> { static const int value =  176; };
template <> struct vnode_shift<2, 3, 30,  4> { static const int value =  0; };
// ILS = 3, CHECK_NODE = 31
template <> struct vnode_shift<2, 3, 31,  0> { static const int value =  147; };
template <> struct vnode_shift<2, 3, 31,  1> { static const int value =  182; };
template <> struct vnode_shift<2, 3, 31,  2> { static const int value =  0; };
// ILS = 3, CHECK_NODE = 32
template <> struct vnode_shift<2, 3, 32,  0> { static const int value =  108; };
template <> struct vnode_shift<2, 3, 32,  1> { static const int value =  21; };
template <> struct vnode_shift<2, 3, 32,  2> { static const int value =  110; };
template <> struct vnode_shift<2, 3, 32,  3> { static const int value =  0; };
// ILS = 3, CHECK_NODE = 33
template <> struct vnode_shift<2, 3, 33,  0> { static const int value =  71; };
template <> struct vnode_shift<2, 3, 33,  1> { static const int value =  12; };
template <> struct vnode_shift<2, 3, 33,  2> { static const int value =  109; };
template <> struct vnode_shift<2, 3, 33,  3> { static const int value =  0; };
// ILS = 3, CHECK_NODE = 34
template <> struct vnode_shift<2, 3, 34,  0> { static const int value =  29; };
template <> struct vnode_shift<2, 3, 34,  1> { static const int value =  201; };
template <> struct vnode_shift<2, 3, 34,  2> { static const int value =  69; };
template <> struct vnode_shift<2, 3, 34,  3> { static const int value =  0; };
// ILS = 3, CHECK_NODE = 35
template <> struct vnode_shift<2, 3, 35,  0> { static const int value =  91; };
template <> struct vnode_shift<2, 3, 35,  1> { static const int value =  165; };
template <> struct vnode_shift<2, 3, 35,  2> { static const int value =  55; };
template <> struct vnode_shift<2, 3, 35,  3> { static const int value =  0; };
// ILS = 3, CHECK_NODE = 36
template <> struct vnode_shift<2, 3, 36,  0> { static const int value =  1; };
template <> struct vnode_shift<2, 3, 36,  1> { static const int value =  175; };
template <> struct vnode_shift<2, 3, 36,  2> { static const int value =  83; };
template <> struct vnode_shift<2, 3, 36,  3> { static const int value =  0; };
// ILS = 3, CHECK_NODE = 37
template <> struct vnode_shift<2, 3, 37,  0> { static const int value =  40; };
template <> struct vnode_shift<2, 3, 37,  1> { static const int value =  12; };
template <> struct vnode_shift<2, 3, 37,  2> { static const int value =  0; };
// ILS = 3, CHECK_NODE = 38
template <> struct vnode_shift<2, 3, 38,  0> { static const int value =  37; };
template <> struct vnode_shift<2, 3, 38,  1> { static const int value =  97; };
template <> struct vnode_shift<2, 3, 38,  2> { static const int value =  46; };
template <> struct vnode_shift<2, 3, 38,  3> { static const int value =  0; };
// ILS = 3, CHECK_NODE = 39
template <> struct vnode_shift<2, 3, 39,  0> { static const int value =  106; };
template <> struct vnode_shift<2, 3, 39,  1> { static const int value =  181; };
template <> struct vnode_shift<2, 3, 39,  2> { static const int value =  154; };
template <> struct vnode_shift<2, 3, 39,  3> { static const int value =  0; };
// ILS = 3, CHECK_NODE = 40
template <> struct vnode_shift<2, 3, 40,  0> { static const int value =  98; };
template <> struct vnode_shift<2, 3, 40,  1> { static const int value =  35; };
template <> struct vnode_shift<2, 3, 40,  2> { static const int value =  36; };
template <> struct vnode_shift<2, 3, 40,  3> { static const int value =  0; };
// ILS = 3, CHECK_NODE = 41
template <> struct vnode_shift<2, 3, 41,  0> { static const int value =  120; };
template <> struct vnode_shift<2, 3, 41,  1> { static const int value =  101; };
template <> struct vnode_shift<2, 3, 41,  2> { static const int value =  81; };
template <> struct vnode_shift<2, 3, 41,  3> { static const int value =  0; };
// ILS = 4, CHECK_NODE = 0
template <> struct vnode_shift<2, 4, 0,  0> { static const int value =  3; };
template <> struct vnode_shift<2, 4, 0,  1> { static const int value =  26; };
template <> struct vnode_shift<2, 4, 0,  2> { static const int value =  53; };
template <> struct vnode_shift<2, 4, 0,  3> { static const int value =  35; };
template <> struct vnode_shift<2, 4, 0,  4> { static const int value =  115; };
template <> struct vnode_shift<2, 4, 0,  5> { static const int value =  127; };
template <> struct vnode_shift<2, 4, 0,  6> { static const int value =  0; };
template <> struct vnode_shift<2, 4, 0,  7> { static const int value =  0; };
// ILS = 4, CHECK_NODE = 1
template <> struct vnode_shift<2, 4, 1,  0> { static const int value =  19; };
template <> struct vnode_shift<2, 4, 1,  1> { static const int value =  94; };
template <> struct vnode_shift<2, 4, 1,  2> { static const int value =  104; };
template <> struct vnode_shift<2, 4, 1,  3> { static const int value =  66; };
template <> struct vnode_shift<2, 4, 1,  4> { static const int value =  84; };
template <> struct vnode_shift<2, 4, 1,  5> { static const int value =  98; };
template <> struct vnode_shift<2, 4, 1,  6> { static const int value =  69; };
template <> struct vnode_shift<2, 4, 1,  7> { static const int value =  50; };
template <> struct vnode_shift<2, 4, 1,  8> { static const int value =  0; };
template <> struct vnode_shift<2, 4, 1,  9> { static const int value =  0; };
// ILS = 4, CHECK_NODE = 2
template <> struct vnode_shift<2, 4, 2,  0> { static const int value =  95; };
template <> struct vnode_shift<2, 4, 2,  1> { static const int value =  106; };
template <> struct vnode_shift<2, 4, 2,  2> { static const int value =  92; };
template <> struct vnode_shift<2, 4, 2,  3> { static const int value =  110; };
template <> struct vnode_shift<2, 4, 2,  4> { static const int value =  111; };
template <> struct vnode_shift<2, 4, 2,  5> { static const int value =  1; };
template <> struct vnode_shift<2, 4, 2,  6> { static const int value =  0; };
template <> struct vnode_shift<2, 4, 2,  7> { static const int value =  0; };
// ILS = 4, CHECK_NODE = 3
template <> struct vnode_shift<2, 4, 3,  0> { static const int value =  120; };
template <> struct vnode_shift<2, 4, 3,  1> { static const int value =  121; };
template <> struct vnode_shift<2, 4, 3,  2> { static const int value =  22; };
template <> struct vnode_shift<2, 4, 3,  3> { static const int value =  4; };
template <> struct vnode_shift<2, 4, 3,  4> { static const int value =  73; };
template <> struct vnode_shift<2, 4, 3,  5> { static const int value =  49; };
template <> struct vnode_shift<2, 4, 3,  6> { static const int value =  128; };
template <> struct vnode_shift<2, 4, 3,  7> { static const int value =  79; };
template <> struct vnode_shift<2, 4, 3,  8> { static const int value =  0; };
template <> struct vnode_shift<2, 4, 3,  9> { static const int value =  0; };
// ILS = 4, CHECK_NODE = 4
template <> struct vnode_shift<2, 4, 4,  0> { static const int value =  42; };
template <> struct vnode_shift<2, 4, 4,  1> { static const int value =  24; };
template <> struct vnode_shift<2, 4, 4,  2> { static const int value =  51; };
template <> struct vnode_shift<2, 4, 4,  3> { static const int value =  0; };
// ILS = 4, CHECK_NODE = 5
template <> struct vnode_shift<2, 4, 5,  0> { static const int value =  40; };
template <> struct vnode_shift<2, 4, 5,  1> { static const int value =  140; };
template <> struct vnode_shift<2, 4, 5,  2> { static const int value =  84; };
template <> struct vnode_shift<2, 4, 5,  3> { static const int value =  137; };
template <> struct vnode_shift<2, 4, 5,  4> { static const int value =  71; };
template <> struct vnode_shift<2, 4, 5,  5> { static const int value =  0; };
// ILS = 4, CHECK_NODE = 6
template <> struct vnode_shift<2, 4, 6,  0> { static const int value =  109; };
template <> struct vnode_shift<2, 4, 6,  1> { static const int value =  87; };
template <> struct vnode_shift<2, 4, 6,  2> { static const int value =  107; };
template <> struct vnode_shift<2, 4, 6,  3> { static const int value =  133; };
template <> struct vnode_shift<2, 4, 6,  4> { static const int value =  139; };
template <> struct vnode_shift<2, 4, 6,  5> { static const int value =  0; };
// ILS = 4, CHECK_NODE = 7
template <> struct vnode_shift<2, 4, 7,  0> { static const int value =  97; };
template <> struct vnode_shift<2, 4, 7,  1> { static const int value =  135; };
template <> struct vnode_shift<2, 4, 7,  2> { static const int value =  35; };
template <> struct vnode_shift<2, 4, 7,  3> { static const int value =  108; };
template <> struct vnode_shift<2, 4, 7,  4> { static const int value =  65; };
template <> struct vnode_shift<2, 4, 7,  5> { static const int value =  0; };
// ILS = 4, CHECK_NODE = 8
template <> struct vnode_shift<2, 4, 8,  0> { static const int value =  70; };
template <> struct vnode_shift<2, 4, 8,  1> { static const int value =  69; };
template <> struct vnode_shift<2, 4, 8,  2> { static const int value =  88; };
template <> struct vnode_shift<2, 4, 8,  3> { static const int value =  0; };
// ILS = 4, CHECK_NODE = 9
template <> struct vnode_shift<2, 4, 9,  0> { static const int value =  97; };
template <> struct vnode_shift<2, 4, 9,  1> { static const int value =  40; };
template <> struct vnode_shift<2, 4, 9,  2> { static const int value =  24; };
template <> struct vnode_shift<2, 4, 9,  3> { static const int value =  49; };
template <> struct vnode_shift<2, 4, 9,  4> { static const int value =  0; };
// ILS = 4, CHECK_NODE = 10
template <> struct vnode_shift<2, 4, 10,  0> { static const int value =  46; };
template <> struct vnode_shift<2, 4, 10,  1> { static const int value =  41; };
template <> struct vnode_shift<2, 4, 10,  2> { static const int value =  101; };
template <> struct vnode_shift<2, 4, 10,  3> { static const int value =  96; };
template <> struct vnode_shift<2, 4, 10,  4> { static const int value =  0; };
// ILS = 4, CHECK_NODE = 11
template <> struct vnode_shift<2, 4, 11,  0> { static const int value =  28; };
template <> struct vnode_shift<2, 4, 11,  1> { static const int value =  30; };
template <> struct vnode_shift<2, 4, 11,  2> { static const int value =  116; };
template <> struct vnode_shift<2, 4, 11,  3> { static const int value =  64; };
template <> struct vnode_shift<2, 4, 11,  4> { static const int value =  0; };
// ILS = 4, CHECK_NODE = 12
template <> struct vnode_shift<2, 4, 12,  0> { static const int value =  33; };
template <> struct vnode_shift<2, 4, 12,  1> { static const int value =  122; };
template <> struct vnode_shift<2, 4, 12,  2> { static const int value =  131; };
template <> struct vnode_shift<2, 4, 12,  3> { static const int value =  0; };
// ILS = 4, CHECK_NODE = 13
template <> struct vnode_shift<2, 4, 13,  0> { static const int value =  76; };
template <> struct vnode_shift<2, 4, 13,  1> { static const int value =  37; };
template <> struct vnode_shift<2, 4, 13,  2> { static const int value =  62; };
template <> struct vnode_shift<2, 4, 13,  3> { static const int value =  47; };
template <> struct vnode_shift<2, 4, 13,  4> { static const int value =  0; };
// ILS = 4, CHECK_NODE = 14
template <> struct vnode_shift<2, 4, 14,  0> { static const int value =  143; };
template <> struct vnode_shift<2, 4, 14,  1> { static const int value =  51; };
template <> struct vnode_shift<2, 4, 14,  2> { static const int value =  130; };
template <> struct vnode_shift<2, 4, 14,  3> { static const int value =  97; };
template <> struct vnode_shift<2, 4, 14,  4> { static const int value =  0; };
// ILS = 4, CHECK_NODE = 15
template <> struct vnode_shift<2, 4, 15,  0> { static const int value =  139; };
template <> struct vnode_shift<2, 4, 15,  1> { static const int value =  96; };
template <> struct vnode_shift<2, 4, 15,  2> { static const int value =  128; };
template <> struct vnode_shift<2, 4, 15,  3> { static const int value =  0; };
// ILS = 4, CHECK_NODE = 16
template <> struct vnode_shift<2, 4, 16,  0> { static const int value =  48; };
template <> struct vnode_shift<2, 4, 16,  1> { static const int value =  9; };
template <> struct vnode_shift<2, 4, 16,  2> { static const int value =  28; };
template <> struct vnode_shift<2, 4, 16,  3> { static const int value =  8; };
template <> struct vnode_shift<2, 4, 16,  4> { static const int value =  0; };
// ILS = 4, CHECK_NODE = 17
template <> struct vnode_shift<2, 4, 17,  0> { static const int value =  120; };
template <> struct vnode_shift<2, 4, 17,  1> { static const int value =  43; };
template <> struct vnode_shift<2, 4, 17,  2> { static const int value =  65; };
template <> struct vnode_shift<2, 4, 17,  3> { static const int value =  42; };
template <> struct vnode_shift<2, 4, 17,  4> { static const int value =  0; };
// ILS = 4, CHECK_NODE = 18
template <> struct vnode_shift<2, 4, 18,  0> { static const int value =  17; };
template <> struct vnode_shift<2, 4, 18,  1> { static const int value =  106; };
template <> struct vnode_shift<2, 4, 18,  2> { static const int value =  142; };
template <> struct vnode_shift<2, 4, 18,  3> { static const int value =  0; };
// ILS = 4, CHECK_NODE = 19
template <> struct vnode_shift<2, 4, 19,  0> { static const int value =  79; };
template <> struct vnode_shift<2, 4, 19,  1> { static const int value =  28; };
template <> struct vnode_shift<2, 4, 19,  2> { static const int value =  41; };
template <> struct vnode_shift<2, 4, 19,  3> { static const int value =  0; };
// ILS = 4, CHECK_NODE = 20
template <> struct vnode_shift<2, 4, 20,  0> { static const int value =  2; };
template <> struct vnode_shift<2, 4, 20,  1> { static const int value =  103; };
template <> struct vnode_shift<2, 4, 20,  2> { static const int value =  78; };
template <> struct vnode_shift<2, 4, 20,  3> { static const int value =  0; };
// ILS = 4, CHECK_NODE = 21
template <> struct vnode_shift<2, 4, 21,  0> { static const int value =  91; };
template <> struct vnode_shift<2, 4, 21,  1> { static const int value =  75; };
template <> struct vnode_shift<2, 4, 21,  2> { static const int value =  81; };
template <> struct vnode_shift<2, 4, 21,  3> { static const int value =  0; };
// ILS = 4, CHECK_NODE = 22
template <> struct vnode_shift<2, 4, 22,  0> { static const int value =  54; };
template <> struct vnode_shift<2, 4, 22,  1> { static const int value =  132; };
template <> struct vnode_shift<2, 4, 22,  2> { static const int value =  0; };
// ILS = 4, CHECK_NODE = 23
template <> struct vnode_shift<2, 4, 23,  0> { static const int value =  68; };
template <> struct vnode_shift<2, 4, 23,  1> { static const int value =  115; };
template <> struct vnode_shift<2, 4, 23,  2> { static const int value =  56; };
template <> struct vnode_shift<2, 4, 23,  3> { static const int value =  0; };
// ILS = 4, CHECK_NODE = 24
template <> struct vnode_shift<2, 4, 24,  0> { static const int value =  30; };
template <> struct vnode_shift<2, 4, 24,  1> { static const int value =  42; };
template <> struct vnode_shift<2, 4, 24,  2> { static const int value =  101; };
template <> struct vnode_shift<2, 4, 24,  3> { static const int value =  0; };
// ILS = 4, CHECK_NODE = 25
template <> struct vnode_shift<2, 4, 25,  0> { static const int value =  128; };
template <> struct vnode_shift<2, 4, 25,  1> { static const int value =  63; };
template <> struct vnode_shift<2, 4, 25,  2> { static const int value =  0; };
// ILS = 4, CHECK_NODE = 26
template <> struct vnode_shift<2, 4, 26,  0> { static const int value =  142; };
template <> struct vnode_shift<2, 4, 26,  1> { static const int value =  28; };
template <> struct vnode_shift<2, 4, 26,  2> { static const int value =  100; };
template <> struct vnode_shift<2, 4, 26,  3> { static const int value =  133; };
template <> struct vnode_shift<2, 4, 26,  4> { static const int value =  0; };
// ILS = 4, CHECK_NODE = 27
template <> struct vnode_shift<2, 4, 27,  0> { static const int value =  13; };
template <> struct vnode_shift<2, 4, 27,  1> { static const int value =  10; };
template <> struct vnode_shift<2, 4, 27,  2> { static const int value =  0; };
// ILS = 4, CHECK_NODE = 28
template <> struct vnode_shift<2, 4, 28,  0> { static const int value =  106; };
template <> struct vnode_shift<2, 4, 28,  1> { static const int value =  77; };
template <> struct vnode_shift<2, 4, 28,  2> { static const int value =  43; };
template <> struct vnode_shift<2, 4, 28,  3> { static const int value =  0; };
// ILS = 4, CHECK_NODE = 29
template <> struct vnode_shift<2, 4, 29,  0> { static const int value =  133; };
template <> struct vnode_shift<2, 4, 29,  1> { static const int value =  25; };
template <> struct vnode_shift<2, 4, 29,  2> { static const int value =  0; };
// ILS = 4, CHECK_NODE = 30
template <> struct vnode_shift<2, 4, 30,  0> { static const int value =  87; };
template <> struct vnode_shift<2, 4, 30,  1> { static const int value =  56; };
template <> struct vnode_shift<2, 4, 30,  2> { static const int value =  104; };
template <> struct vnode_shift<2, 4, 30,  3> { static const int value =  70; };
template <> struct vnode_shift<2, 4, 30,  4> { static const int value =  0; };
// ILS = 4, CHECK_NODE = 31
template <> struct vnode_shift<2, 4, 31,  0> { static const int value =  80; };
template <> struct vnode_shift<2, 4, 31,  1> { static const int value =  139; };
template <> struct vnode_shift<2, 4, 31,  2> { static const int value =  0; };
// ILS = 4, CHECK_NODE = 32
template <> struct vnode_shift<2, 4, 32,  0> { static const int value =  32; };
template <> struct vnode_shift<2, 4, 32,  1> { static const int value =  89; };
template <> struct vnode_shift<2, 4, 32,  2> { static const int value =  71; };
template <> struct vnode_shift<2, 4, 32,  3> { static const int value =  0; };
// ILS = 4, CHECK_NODE = 33
template <> struct vnode_shift<2, 4, 33,  0> { static const int value =  135; };
template <> struct vnode_shift<2, 4, 33,  1> { static const int value =  6; };
template <> struct vnode_shift<2, 4, 33,  2> { static const int value =  2; };
template <> struct vnode_shift<2, 4, 33,  3> { static const int value =  0; };
// ILS = 4, CHECK_NODE = 34
template <> struct vnode_shift<2, 4, 34,  0> { static const int value =  37; };
template <> struct vnode_shift<2, 4, 34,  1> { static const int value =  25; };
template <> struct vnode_shift<2, 4, 34,  2> { static const int value =  114; };
template <> struct vnode_shift<2, 4, 34,  3> { static const int value =  0; };
// ILS = 4, CHECK_NODE = 35
template <> struct vnode_shift<2, 4, 35,  0> { static const int value =  60; };
template <> struct vnode_shift<2, 4, 35,  1> { static const int value =  137; };
template <> struct vnode_shift<2, 4, 35,  2> { static const int value =  93; };
template <> struct vnode_shift<2, 4, 35,  3> { static const int value =  0; };
// ILS = 4, CHECK_NODE = 36
template <> struct vnode_shift<2, 4, 36,  0> { static const int value =  121; };
template <> struct vnode_shift<2, 4, 36,  1> { static const int value =  129; };
template <> struct vnode_shift<2, 4, 36,  2> { static const int value =  26; };
template <> struct vnode_shift<2, 4, 36,  3> { static const int value =  0; };
// ILS = 4, CHECK_NODE = 37
template <> struct vnode_shift<2, 4, 37,  0> { static const int value =  97; };
template <> struct vnode_shift<2, 4, 37,  1> { static const int value =  56; };
template <> struct vnode_shift<2, 4, 37,  2> { static const int value =  0; };
// ILS = 4, CHECK_NODE = 38
template <> struct vnode_shift<2, 4, 38,  0> { static const int value =  1; };
template <> struct vnode_shift<2, 4, 38,  1> { static const int value =  70; };
template <> struct vnode_shift<2, 4, 38,  2> { static const int value =  1; };
template <> struct vnode_shift<2, 4, 38,  3> { static const int value =  0; };
// ILS = 4, CHECK_NODE = 39
template <> struct vnode_shift<2, 4, 39,  0> { static const int value =  119; };
template <> struct vnode_shift<2, 4, 39,  1> { static const int value =  32; };
template <> struct vnode_shift<2, 4, 39,  2> { static const int value =  142; };
template <> struct vnode_shift<2, 4, 39,  3> { static const int value =  0; };
// ILS = 4, CHECK_NODE = 40
template <> struct vnode_shift<2, 4, 40,  0> { static const int value =  6; };
template <> struct vnode_shift<2, 4, 40,  1> { static const int value =  73; };
template <> struct vnode_shift<2, 4, 40,  2> { static const int value =  102; };
template <> struct vnode_shift<2, 4, 40,  3> { static const int value =  0; };
// ILS = 4, CHECK_NODE = 41
template <> struct vnode_shift<2, 4, 41,  0> { static const int value =  48; };
template <> struct vnode_shift<2, 4, 41,  1> { static const int value =  47; };
template <> struct vnode_shift<2, 4, 41,  2> { static const int value =  19; };
template <> struct vnode_shift<2, 4, 41,  3> { static const int value =  0; };
// ILS = 5, CHECK_NODE = 0
template <> struct vnode_shift<2, 5, 0,  0> { static const int value =  156; };
template <> struct vnode_shift<2, 5, 0,  1> { static const int value =  143; };
template <> struct vnode_shift<2, 5, 0,  2> { static const int value =  14; };
template <> struct vnode_shift<2, 5, 0,  3> { static const int value =  3; };
template <> struct vnode_shift<2, 5, 0,  4> { static const int value =  40; };
template <> struct vnode_shift<2, 5, 0,  5> { static const int value =  123; };
template <> struct vnode_shift<2, 5, 0,  6> { static const int value =  0; };
template <> struct vnode_shift<2, 5, 0,  7> { static const int value =  0; };
// ILS = 5, CHECK_NODE = 1
template <> struct vnode_shift<2, 5, 1,  0> { static const int value =  17; };
template <> struct vnode_shift<2, 5, 1,  1> { static const int value =  65; };
template <> struct vnode_shift<2, 5, 1,  2> { static const int value =  63; };
template <> struct vnode_shift<2, 5, 1,  3> { static const int value =  1; };
template <> struct vnode_shift<2, 5, 1,  4> { static const int value =  55; };
template <> struct vnode_shift<2, 5, 1,  5> { static const int value =  37; };
template <> struct vnode_shift<2, 5, 1,  6> { static const int value =  171; };
template <> struct vnode_shift<2, 5, 1,  7> { static const int value =  133; };
template <> struct vnode_shift<2, 5, 1,  8> { static const int value =  0; };
template <> struct vnode_shift<2, 5, 1,  9> { static const int value =  0; };
// ILS = 5, CHECK_NODE = 2
template <> struct vnode_shift<2, 5, 2,  0> { static const int value =  98; };
template <> struct vnode_shift<2, 5, 2,  1> { static const int value =  168; };
template <> struct vnode_shift<2, 5, 2,  2> { static const int value =  107; };
template <> struct vnode_shift<2, 5, 2,  3> { static const int value =  82; };
template <> struct vnode_shift<2, 5, 2,  4> { static const int value =  142; };
template <> struct vnode_shift<2, 5, 2,  5> { static const int value =  1; };
template <> struct vnode_shift<2, 5, 2,  6> { static const int value =  0; };
template <> struct vnode_shift<2, 5, 2,  7> { static const int value =  0; };
// ILS = 5, CHECK_NODE = 3
template <> struct vnode_shift<2, 5, 3,  0> { static const int value =  53; };
template <> struct vnode_shift<2, 5, 3,  1> { static const int value =  174; };
template <> struct vnode_shift<2, 5, 3,  2> { static const int value =  174; };
template <> struct vnode_shift<2, 5, 3,  3> { static const int value =  127; };
template <> struct vnode_shift<2, 5, 3,  4> { static const int value =  17; };
template <> struct vnode_shift<2, 5, 3,  5> { static const int value =  89; };
template <> struct vnode_shift<2, 5, 3,  6> { static const int value =  17; };
template <> struct vnode_shift<2, 5, 3,  7> { static const int value =  105; };
template <> struct vnode_shift<2, 5, 3,  8> { static const int value =  0; };
template <> struct vnode_shift<2, 5, 3,  9> { static const int value =  0; };
// ILS = 5, CHECK_NODE = 4
template <> struct vnode_shift<2, 5, 4,  0> { static const int value =  86; };
template <> struct vnode_shift<2, 5, 4,  1> { static const int value =  67; };
template <> struct vnode_shift<2, 5, 4,  2> { static const int value =  83; };
template <> struct vnode_shift<2, 5, 4,  3> { static const int value =  0; };
// ILS = 5, CHECK_NODE = 5
template <> struct vnode_shift<2, 5, 5,  0> { static const int value =  79; };
template <> struct vnode_shift<2, 5, 5,  1> { static const int value =  84; };
template <> struct vnode_shift<2, 5, 5,  2> { static const int value =  35; };
template <> struct vnode_shift<2, 5, 5,  3> { static const int value =  103; };
template <> struct vnode_shift<2, 5, 5,  4> { static const int value =  60; };
template <> struct vnode_shift<2, 5, 5,  5> { static const int value =  0; };
// ILS = 5, CHECK_NODE = 6
template <> struct vnode_shift<2, 5, 6,  0> { static const int value =  47; };
template <> struct vnode_shift<2, 5, 6,  1> { static const int value =  154; };
template <> struct vnode_shift<2, 5, 6,  2> { static const int value =  10; };
template <> struct vnode_shift<2, 5, 6,  3> { static const int value =  155; };
template <> struct vnode_shift<2, 5, 6,  4> { static const int value =  29; };
template <> struct vnode_shift<2, 5, 6,  5> { static const int value =  0; };
// ILS = 5, CHECK_NODE = 7
template <> struct vnode_shift<2, 5, 7,  0> { static const int value =  48; };
template <> struct vnode_shift<2, 5, 7,  1> { static const int value =  125; };
template <> struct vnode_shift<2, 5, 7,  2> { static const int value =  24; };
template <> struct vnode_shift<2, 5, 7,  3> { static const int value =  47; };
template <> struct vnode_shift<2, 5, 7,  4> { static const int value =  55; };
template <> struct vnode_shift<2, 5, 7,  5> { static const int value =  0; };
// ILS = 5, CHECK_NODE = 8
template <> struct vnode_shift<2, 5, 8,  0> { static const int value =  53; };
template <> struct vnode_shift<2, 5, 8,  1> { static const int value =  31; };
template <> struct vnode_shift<2, 5, 8,  2> { static const int value =  161; };
template <> struct vnode_shift<2, 5, 8,  3> { static const int value =  0; };
// ILS = 5, CHECK_NODE = 9
template <> struct vnode_shift<2, 5, 9,  0> { static const int value =  104; };
template <> struct vnode_shift<2, 5, 9,  1> { static const int value =  142; };
template <> struct vnode_shift<2, 5, 9,  2> { static const int value =  99; };
template <> struct vnode_shift<2, 5, 9,  3> { static const int value =  64; };
template <> struct vnode_shift<2, 5, 9,  4> { static const int value =  0; };
// ILS = 5, CHECK_NODE = 10
template <> struct vnode_shift<2, 5, 10,  0> { static const int value =  111; };
template <> struct vnode_shift<2, 5, 10,  1> { static const int value =  25; };
template <> struct vnode_shift<2, 5, 10,  2> { static const int value =  174; };
template <> struct vnode_shift<2, 5, 10,  3> { static const int value =  23; };
template <> struct vnode_shift<2, 5, 10,  4> { static const int value =  0; };
// ILS = 5, CHECK_NODE = 11
template <> struct vnode_shift<2, 5, 11,  0> { static const int value =  91; };
template <> struct vnode_shift<2, 5, 11,  1> { static const int value =  175; };
template <> struct vnode_shift<2, 5, 11,  2> { static const int value =  24; };
template <> struct vnode_shift<2, 5, 11,  3> { static const int value =  141; };
template <> struct vnode_shift<2, 5, 11,  4> { static const int value =  0; };
// ILS = 5, CHECK_NODE = 12
template <> struct vnode_shift<2, 5, 12,  0> { static const int value =  122; };
template <> struct vnode_shift<2, 5, 12,  1> { static const int value =  11; };
template <> struct vnode_shift<2, 5, 12,  2> { static const int value =  4; };
template <> struct vnode_shift<2, 5, 12,  3> { static const int value =  0; };
// ILS = 5, CHECK_NODE = 13
template <> struct vnode_shift<2, 5, 13,  0> { static const int value =  29; };
template <> struct vnode_shift<2, 5, 13,  1> { static const int value =  91; };
template <> struct vnode_shift<2, 5, 13,  2> { static const int value =  27; };
template <> struct vnode_shift<2, 5, 13,  3> { static const int value =  127; };
template <> struct vnode_shift<2, 5, 13,  4> { static const int value =  0; };
// ILS = 5, CHECK_NODE = 14
template <> struct vnode_shift<2, 5, 14,  0> { static const int value =  11; };
template <> struct vnode_shift<2, 5, 14,  1> { static const int value =  145; };
template <> struct vnode_shift<2, 5, 14,  2> { static const int value =  8; };
template <> struct vnode_shift<2, 5, 14,  3> { static const int value =  166; };
template <> struct vnode_shift<2, 5, 14,  4> { static const int value =  0; };
// ILS = 5, CHECK_NODE = 15
template <> struct vnode_shift<2, 5, 15,  0> { static const int value =  137; };
template <> struct vnode_shift<2, 5, 15,  1> { static const int value =  103; };
template <> struct vnode_shift<2, 5, 15,  2> { static const int value =  40; };
template <> struct vnode_shift<2, 5, 15,  3> { static const int value =  0; };
// ILS = 5, CHECK_NODE = 16
template <> struct vnode_shift<2, 5, 16,  0> { static const int value =  78; };
template <> struct vnode_shift<2, 5, 16,  1> { static const int value =  158; };
template <> struct vnode_shift<2, 5, 16,  2> { static const int value =  17; };
template <> struct vnode_shift<2, 5, 16,  3> { static const int value =  165; };
template <> struct vnode_shift<2, 5, 16,  4> { static const int value =  0; };
// ILS = 5, CHECK_NODE = 17
template <> struct vnode_shift<2, 5, 17,  0> { static const int value =  134; };
template <> struct vnode_shift<2, 5, 17,  1> { static const int value =  23; };
template <> struct vnode_shift<2, 5, 17,  2> { static const int value =  62; };
template <> struct vnode_shift<2, 5, 17,  3> { static const int value =  163; };
template <> struct vnode_shift<2, 5, 17,  4> { static const int value =  0; };
// ILS = 5, CHECK_NODE = 18
template <> struct vnode_shift<2, 5, 18,  0> { static const int value =  173; };
template <> struct vnode_shift<2, 5, 18,  1> { static const int value =  31; };
template <> struct vnode_shift<2, 5, 18,  2> { static const int value =  22; };
template <> struct vnode_shift<2, 5, 18,  3> { static const int value =  0; };
// ILS = 5, CHECK_NODE = 19
template <> struct vnode_shift<2, 5, 19,  0> { static const int value =  13; };
template <> struct vnode_shift<2, 5, 19,  1> { static const int value =  135; };
template <> struct vnode_shift<2, 5, 19,  2> { static const int value =  145; };
template <> struct vnode_shift<2, 5, 19,  3> { static const int value =  0; };
// ILS = 5, CHECK_NODE = 20
template <> struct vnode_shift<2, 5, 20,  0> { static const int value =  128; };
template <> struct vnode_shift<2, 5, 20,  1> { static const int value =  52; };
template <> struct vnode_shift<2, 5, 20,  2> { static const int value =  173; };
template <> struct vnode_shift<2, 5, 20,  3> { static const int value =  0; };
// ILS = 5, CHECK_NODE = 21
template <> struct vnode_shift<2, 5, 21,  0> { static const int value =  156; };
template <> struct vnode_shift<2, 5, 21,  1> { static const int value =  166; };
template <> struct vnode_shift<2, 5, 21,  2> { static const int value =  40; };
template <> struct vnode_shift<2, 5, 21,  3> { static const int value =  0; };
// ILS = 5, CHECK_NODE = 22
template <> struct vnode_shift<2, 5, 22,  0> { static const int value =  18; };
template <> struct vnode_shift<2, 5, 22,  1> { static const int value =  163; };
template <> struct vnode_shift<2, 5, 22,  2> { static const int value =  0; };
// ILS = 5, CHECK_NODE = 23
template <> struct vnode_shift<2, 5, 23,  0> { static const int value =  110; };
template <> struct vnode_shift<2, 5, 23,  1> { static const int value =  132; };
template <> struct vnode_shift<2, 5, 23,  2> { static const int value =  150; };
template <> struct vnode_shift<2, 5, 23,  3> { static const int value =  0; };
// ILS = 5, CHECK_NODE = 24
template <> struct vnode_shift<2, 5, 24,  0> { static const int value =  113; };
template <> struct vnode_shift<2, 5, 24,  1> { static const int value =  108; };
template <> struct vnode_shift<2, 5, 24,  2> { static const int value =  61; };
template <> struct vnode_shift<2, 5, 24,  3> { static const int value =  0; };
// ILS = 5, CHECK_NODE = 25
template <> struct vnode_shift<2, 5, 25,  0> { static const int value =  72; };
template <> struct vnode_shift<2, 5, 25,  1> { static const int value =  136; };
template <> struct vnode_shift<2, 5, 25,  2> { static const int value =  0; };
// ILS = 5, CHECK_NODE = 26
template <> struct vnode_shift<2, 5, 26,  0> { static const int value =  36; };
template <> struct vnode_shift<2, 5, 26,  1> { static const int value =  38; };
template <> struct vnode_shift<2, 5, 26,  2> { static const int value =  53; };
template <> struct vnode_shift<2, 5, 26,  3> { static const int value =  145; };
template <> struct vnode_shift<2, 5, 26,  4> { static const int value =  0; };
// ILS = 5, CHECK_NODE = 27
template <> struct vnode_shift<2, 5, 27,  0> { static const int value =  42; };
template <> struct vnode_shift<2, 5, 27,  1> { static const int value =  104; };
template <> struct vnode_shift<2, 5, 27,  2> { static const int value =  0; };
// ILS = 5, CHECK_NODE = 28
template <> struct vnode_shift<2, 5, 28,  0> { static const int value =  64; };
template <> struct vnode_shift<2, 5, 28,  1> { static const int value =  24; };
template <> struct vnode_shift<2, 5, 28,  2> { static const int value =  149; };
template <> struct vnode_shift<2, 5, 28,  3> { static const int value =  0; };
// ILS = 5, CHECK_NODE = 29
template <> struct vnode_shift<2, 5, 29,  0> { static const int value =  139; };
template <> struct vnode_shift<2, 5, 29,  1> { static const int value =  161; };
template <> struct vnode_shift<2, 5, 29,  2> { static const int value =  0; };
// ILS = 5, CHECK_NODE = 30
template <> struct vnode_shift<2, 5, 30,  0> { static const int value =  84; };
template <> struct vnode_shift<2, 5, 30,  1> { static const int value =  173; };
template <> struct vnode_shift<2, 5, 30,  2> { static const int value =  93; };
template <> struct vnode_shift<2, 5, 30,  3> { static const int value =  29; };
template <> struct vnode_shift<2, 5, 30,  4> { static const int value =  0; };
// ILS = 5, CHECK_NODE = 31
template <> struct vnode_shift<2, 5, 31,  0> { static const int value =  117; };
template <> struct vnode_shift<2, 5, 31,  1> { static const int value =  148; };
template <> struct vnode_shift<2, 5, 31,  2> { static const int value =  0; };
// ILS = 5, CHECK_NODE = 32
template <> struct vnode_shift<2, 5, 32,  0> { static const int value =  116; };
template <> struct vnode_shift<2, 5, 32,  1> { static const int value =  73; };
template <> struct vnode_shift<2, 5, 32,  2> { static const int value =  142; };
template <> struct vnode_shift<2, 5, 32,  3> { static const int value =  0; };
// ILS = 5, CHECK_NODE = 33
template <> struct vnode_shift<2, 5, 33,  0> { static const int value =  105; };
template <> struct vnode_shift<2, 5, 33,  1> { static const int value =  137; };
template <> struct vnode_shift<2, 5, 33,  2> { static const int value =  29; };
template <> struct vnode_shift<2, 5, 33,  3> { static const int value =  0; };
// ILS = 5, CHECK_NODE = 34
template <> struct vnode_shift<2, 5, 34,  0> { static const int value =  11; };
template <> struct vnode_shift<2, 5, 34,  1> { static const int value =  41; };
template <> struct vnode_shift<2, 5, 34,  2> { static const int value =  162; };
template <> struct vnode_shift<2, 5, 34,  3> { static const int value =  0; };
// ILS = 5, CHECK_NODE = 35
template <> struct vnode_shift<2, 5, 35,  0> { static const int value =  126; };
template <> struct vnode_shift<2, 5, 35,  1> { static const int value =  152; };
template <> struct vnode_shift<2, 5, 35,  2> { static const int value =  172; };
template <> struct vnode_shift<2, 5, 35,  3> { static const int value =  0; };
// ILS = 5, CHECK_NODE = 36
template <> struct vnode_shift<2, 5, 36,  0> { static const int value =  73; };
template <> struct vnode_shift<2, 5, 36,  1> { static const int value =  154; };
template <> struct vnode_shift<2, 5, 36,  2> { static const int value =  129; };
template <> struct vnode_shift<2, 5, 36,  3> { static const int value =  0; };
// ILS = 5, CHECK_NODE = 37
template <> struct vnode_shift<2, 5, 37,  0> { static const int value =  167; };
template <> struct vnode_shift<2, 5, 37,  1> { static const int value =  38; };
template <> struct vnode_shift<2, 5, 37,  2> { static const int value =  0; };
// ILS = 5, CHECK_NODE = 38
template <> struct vnode_shift<2, 5, 38,  0> { static const int value =  112; };
template <> struct vnode_shift<2, 5, 38,  1> { static const int value =  7; };
template <> struct vnode_shift<2, 5, 38,  2> { static const int value =  19; };
template <> struct vnode_shift<2, 5, 38,  3> { static const int value =  0; };
// ILS = 5, CHECK_NODE = 39
template <> struct vnode_shift<2, 5, 39,  0> { static const int value =  109; };
template <> struct vnode_shift<2, 5, 39,  1> { static const int value =  6; };
template <> struct vnode_shift<2, 5, 39,  2> { static const int value =  105; };
template <> struct vnode_shift<2, 5, 39,  3> { static const int value =  0; };
// ILS = 5, CHECK_NODE = 40
template <> struct vnode_shift<2, 5, 40,  0> { static const int value =  160; };
template <> struct vnode_shift<2, 5, 40,  1> { static const int value =  156; };
template <> struct vnode_shift<2, 5, 40,  2> { static const int value =  82; };
template <> struct vnode_shift<2, 5, 40,  3> { static const int value =  0; };
// ILS = 5, CHECK_NODE = 41
template <> struct vnode_shift<2, 5, 41,  0> { static const int value =  132; };
template <> struct vnode_shift<2, 5, 41,  1> { static const int value =  6; };
template <> struct vnode_shift<2, 5, 41,  2> { static const int value =  8; };
template <> struct vnode_shift<2, 5, 41,  3> { static const int value =  0; };
// ILS = 6, CHECK_NODE = 0
template <> struct vnode_shift<2, 6, 0,  0> { static const int value =  143; };
template <> struct vnode_shift<2, 6, 0,  1> { static const int value =  19; };
template <> struct vnode_shift<2, 6, 0,  2> { static const int value =  176; };
template <> struct vnode_shift<2, 6, 0,  3> { static const int value =  165; };
template <> struct vnode_shift<2, 6, 0,  4> { static const int value =  196; };
template <> struct vnode_shift<2, 6, 0,  5> { static const int value =  13; };
template <> struct vnode_shift<2, 6, 0,  6> { static const int value =  0; };
template <> struct vnode_shift<2, 6, 0,  7> { static const int value =  0; };
// ILS = 6, CHECK_NODE = 1
template <> struct vnode_shift<2, 6, 1,  0> { static const int value =  18; };
template <> struct vnode_shift<2, 6, 1,  1> { static const int value =  27; };
template <> struct vnode_shift<2, 6, 1,  2> { static const int value =  3; };
template <> struct vnode_shift<2, 6, 1,  3> { static const int value =  102; };
template <> struct vnode_shift<2, 6, 1,  4> { static const int value =  185; };
template <> struct vnode_shift<2, 6, 1,  5> { static const int value =  17; };
template <> struct vnode_shift<2, 6, 1,  6> { static const int value =  14; };
template <> struct vnode_shift<2, 6, 1,  7> { static const int value =  180; };
template <> struct vnode_shift<2, 6, 1,  8> { static const int value =  0; };
template <> struct vnode_shift<2, 6, 1,  9> { static const int value =  0; };
// ILS = 6, CHECK_NODE = 2
template <> struct vnode_shift<2, 6, 2,  0> { static const int value =  126; };
template <> struct vnode_shift<2, 6, 2,  1> { static const int value =  163; };
template <> struct vnode_shift<2, 6, 2,  2> { static const int value =  47; };
template <> struct vnode_shift<2, 6, 2,  3> { static const int value =  183; };
template <> struct vnode_shift<2, 6, 2,  4> { static const int value =  132; };
template <> struct vnode_shift<2, 6, 2,  5> { static const int value =  1; };
template <> struct vnode_shift<2, 6, 2,  6> { static const int value =  0; };
template <> struct vnode_shift<2, 6, 2,  7> { static const int value =  0; };
// ILS = 6, CHECK_NODE = 3
template <> struct vnode_shift<2, 6, 3,  0> { static const int value =  36; };
template <> struct vnode_shift<2, 6, 3,  1> { static const int value =  48; };
template <> struct vnode_shift<2, 6, 3,  2> { static const int value =  18; };
template <> struct vnode_shift<2, 6, 3,  3> { static const int value =  111; };
template <> struct vnode_shift<2, 6, 3,  4> { static const int value =  203; };
template <> struct vnode_shift<2, 6, 3,  5> { static const int value =  3; };
template <> struct vnode_shift<2, 6, 3,  6> { static const int value =  191; };
template <> struct vnode_shift<2, 6, 3,  7> { static const int value =  160; };
template <> struct vnode_shift<2, 6, 3,  8> { static const int value =  0; };
template <> struct vnode_shift<2, 6, 3,  9> { static const int value =  0; };
// ILS = 6, CHECK_NODE = 4
template <> struct vnode_shift<2, 6, 4,  0> { static const int value =  43; };
template <> struct vnode_shift<2, 6, 4,  1> { static const int value =  27; };
template <> struct vnode_shift<2, 6, 4,  2> { static const int value =  117; };
template <> struct vnode_shift<2, 6, 4,  3> { static const int value =  0; };
// ILS = 6, CHECK_NODE = 5
template <> struct vnode_shift<2, 6, 5,  0> { static const int value =  136; };
template <> struct vnode_shift<2, 6, 5,  1> { static const int value =  49; };
template <> struct vnode_shift<2, 6, 5,  2> { static const int value =  36; };
template <> struct vnode_shift<2, 6, 5,  3> { static const int value =  132; };
template <> struct vnode_shift<2, 6, 5,  4> { static const int value =  62; };
template <> struct vnode_shift<2, 6, 5,  5> { static const int value =  0; };
// ILS = 6, CHECK_NODE = 6
template <> struct vnode_shift<2, 6, 6,  0> { static const int value =  7; };
template <> struct vnode_shift<2, 6, 6,  1> { static const int value =  34; };
template <> struct vnode_shift<2, 6, 6,  2> { static const int value =  198; };
template <> struct vnode_shift<2, 6, 6,  3> { static const int value =  168; };
template <> struct vnode_shift<2, 6, 6,  4> { static const int value =  12; };
template <> struct vnode_shift<2, 6, 6,  5> { static const int value =  0; };
// ILS = 6, CHECK_NODE = 7
template <> struct vnode_shift<2, 6, 7,  0> { static const int value =  163; };
template <> struct vnode_shift<2, 6, 7,  1> { static const int value =  78; };
template <> struct vnode_shift<2, 6, 7,  2> { static const int value =  143; };
template <> struct vnode_shift<2, 6, 7,  3> { static const int value =  107; };
template <> struct vnode_shift<2, 6, 7,  4> { static const int value =  58; };
template <> struct vnode_shift<2, 6, 7,  5> { static const int value =  0; };
// ILS = 6, CHECK_NODE = 8
template <> struct vnode_shift<2, 6, 8,  0> { static const int value =  101; };
template <> struct vnode_shift<2, 6, 8,  1> { static const int value =  177; };
template <> struct vnode_shift<2, 6, 8,  2> { static const int value =  22; };
template <> struct vnode_shift<2, 6, 8,  3> { static const int value =  0; };
// ILS = 6, CHECK_NODE = 9
template <> struct vnode_shift<2, 6, 9,  0> { static const int value =  186; };
template <> struct vnode_shift<2, 6, 9,  1> { static const int value =  27; };
template <> struct vnode_shift<2, 6, 9,  2> { static const int value =  205; };
template <> struct vnode_shift<2, 6, 9,  3> { static const int value =  81; };
template <> struct vnode_shift<2, 6, 9,  4> { static const int value =  0; };
// ILS = 6, CHECK_NODE = 10
template <> struct vnode_shift<2, 6, 10,  0> { static const int value =  125; };
template <> struct vnode_shift<2, 6, 10,  1> { static const int value =  60; };
template <> struct vnode_shift<2, 6, 10,  2> { static const int value =  177; };
template <> struct vnode_shift<2, 6, 10,  3> { static const int value =  51; };
template <> struct vnode_shift<2, 6, 10,  4> { static const int value =  0; };
// ILS = 6, CHECK_NODE = 11
template <> struct vnode_shift<2, 6, 11,  0> { static const int value =  39; };
template <> struct vnode_shift<2, 6, 11,  1> { static const int value =  29; };
template <> struct vnode_shift<2, 6, 11,  2> { static const int value =  35; };
template <> struct vnode_shift<2, 6, 11,  3> { static const int value =  8; };
template <> struct vnode_shift<2, 6, 11,  4> { static const int value =  0; };
// ILS = 6, CHECK_NODE = 12
template <> struct vnode_shift<2, 6, 12,  0> { static const int value =  18; };
template <> struct vnode_shift<2, 6, 12,  1> { static const int value =  155; };
template <> struct vnode_shift<2, 6, 12,  2> { static const int value =  49; };
template <> struct vnode_shift<2, 6, 12,  3> { static const int value =  0; };
// ILS = 6, CHECK_NODE = 13
template <> struct vnode_shift<2, 6, 13,  0> { static const int value =  32; };
template <> struct vnode_shift<2, 6, 13,  1> { static const int value =  53; };
template <> struct vnode_shift<2, 6, 13,  2> { static const int value =  95; };
template <> struct vnode_shift<2, 6, 13,  3> { static const int value =  186; };
template <> struct vnode_shift<2, 6, 13,  4> { static const int value =  0; };
// ILS = 6, CHECK_NODE = 14
template <> struct vnode_shift<2, 6, 14,  0> { static const int value =  91; };
template <> struct vnode_shift<2, 6, 14,  1> { static const int value =  20; };
template <> struct vnode_shift<2, 6, 14,  2> { static const int value =  52; };
template <> struct vnode_shift<2, 6, 14,  3> { static const int value =  109; };
template <> struct vnode_shift<2, 6, 14,  4> { static const int value =  0; };
// ILS = 6, CHECK_NODE = 15
template <> struct vnode_shift<2, 6, 15,  0> { static const int value =  174; };
template <> struct vnode_shift<2, 6, 15,  1> { static const int value =  108; };
template <> struct vnode_shift<2, 6, 15,  2> { static const int value =  102; };
template <> struct vnode_shift<2, 6, 15,  3> { static const int value =  0; };
// ILS = 6, CHECK_NODE = 16
template <> struct vnode_shift<2, 6, 16,  0> { static const int value =  125; };
template <> struct vnode_shift<2, 6, 16,  1> { static const int value =  31; };
template <> struct vnode_shift<2, 6, 16,  2> { static const int value =  54; };
template <> struct vnode_shift<2, 6, 16,  3> { static const int value =  176; };
template <> struct vnode_shift<2, 6, 16,  4> { static const int value =  0; };
// ILS = 6, CHECK_NODE = 17
template <> struct vnode_shift<2, 6, 17,  0> { static const int value =  57; };
template <> struct vnode_shift<2, 6, 17,  1> { static const int value =  201; };
template <> struct vnode_shift<2, 6, 17,  2> { static const int value =  142; };
template <> struct vnode_shift<2, 6, 17,  3> { static const int value =  35; };
template <> struct vnode_shift<2, 6, 17,  4> { static const int value =  0; };
// ILS = 6, CHECK_NODE = 18
template <> struct vnode_shift<2, 6, 18,  0> { static const int value =  129; };
template <> struct vnode_shift<2, 6, 18,  1> { static const int value =  203; };
template <> struct vnode_shift<2, 6, 18,  2> { static const int value =  140; };
template <> struct vnode_shift<2, 6, 18,  3> { static const int value =  0; };
// ILS = 6, CHECK_NODE = 19
template <> struct vnode_shift<2, 6, 19,  0> { static const int value =  110; };
template <> struct vnode_shift<2, 6, 19,  1> { static const int value =  124; };
template <> struct vnode_shift<2, 6, 19,  2> { static const int value =  52; };
template <> struct vnode_shift<2, 6, 19,  3> { static const int value =  0; };
// ILS = 6, CHECK_NODE = 20
template <> struct vnode_shift<2, 6, 20,  0> { static const int value =  196; };
template <> struct vnode_shift<2, 6, 20,  1> { static const int value =  35; };
template <> struct vnode_shift<2, 6, 20,  2> { static const int value =  114; };
template <> struct vnode_shift<2, 6, 20,  3> { static const int value =  0; };
// ILS = 6, CHECK_NODE = 21
template <> struct vnode_shift<2, 6, 21,  0> { static const int value =  10; };
template <> struct vnode_shift<2, 6, 21,  1> { static const int value =  122; };
template <> struct vnode_shift<2, 6, 21,  2> { static const int value =  23; };
template <> struct vnode_shift<2, 6, 21,  3> { static const int value =  0; };
// ILS = 6, CHECK_NODE = 22
template <> struct vnode_shift<2, 6, 22,  0> { static const int value =  202; };
template <> struct vnode_shift<2, 6, 22,  1> { static const int value =  126; };
template <> struct vnode_shift<2, 6, 22,  2> { static const int value =  0; };
// ILS = 6, CHECK_NODE = 23
template <> struct vnode_shift<2, 6, 23,  0> { static const int value =  52; };
template <> struct vnode_shift<2, 6, 23,  1> { static const int value =  170; };
template <> struct vnode_shift<2, 6, 23,  2> { static const int value =  13; };
template <> struct vnode_shift<2, 6, 23,  3> { static const int value =  0; };
// ILS = 6, CHECK_NODE = 24
template <> struct vnode_shift<2, 6, 24,  0> { static const int value =  113; };
template <> struct vnode_shift<2, 6, 24,  1> { static const int value =  161; };
template <> struct vnode_shift<2, 6, 24,  2> { static const int value =  88; };
template <> struct vnode_shift<2, 6, 24,  3> { static const int value =  0; };
// ILS = 6, CHECK_NODE = 25
template <> struct vnode_shift<2, 6, 25,  0> { static const int value =  197; };
template <> struct vnode_shift<2, 6, 25,  1> { static const int value =  194; };
template <> struct vnode_shift<2, 6, 25,  2> { static const int value =  0; };
// ILS = 6, CHECK_NODE = 26
template <> struct vnode_shift<2, 6, 26,  0> { static const int value =  164; };
template <> struct vnode_shift<2, 6, 26,  1> { static const int value =  172; };
template <> struct vnode_shift<2, 6, 26,  2> { static const int value =  49; };
template <> struct vnode_shift<2, 6, 26,  3> { static const int value =  161; };
template <> struct vnode_shift<2, 6, 26,  4> { static const int value =  0; };
// ILS = 6, CHECK_NODE = 27
template <> struct vnode_shift<2, 6, 27,  0> { static const int value =  168; };
template <> struct vnode_shift<2, 6, 27,  1> { static const int value =  193; };
template <> struct vnode_shift<2, 6, 27,  2> { static const int value =  0; };
// ILS = 6, CHECK_NODE = 28
template <> struct vnode_shift<2, 6, 28,  0> { static const int value =  14; };
template <> struct vnode_shift<2, 6, 28,  1> { static const int value =  186; };
template <> struct vnode_shift<2, 6, 28,  2> { static const int value =  46; };
template <> struct vnode_shift<2, 6, 28,  3> { static const int value =  0; };
// ILS = 6, CHECK_NODE = 29
template <> struct vnode_shift<2, 6, 29,  0> { static const int value =  50; };
template <> struct vnode_shift<2, 6, 29,  1> { static const int value =  27; };
template <> struct vnode_shift<2, 6, 29,  2> { static const int value =  0; };
// ILS = 6, CHECK_NODE = 30
template <> struct vnode_shift<2, 6, 30,  0> { static const int value =  70; };
template <> struct vnode_shift<2, 6, 30,  1> { static const int value =  17; };
template <> struct vnode_shift<2, 6, 30,  2> { static const int value =  50; };
template <> struct vnode_shift<2, 6, 30,  3> { static const int value =  6; };
template <> struct vnode_shift<2, 6, 30,  4> { static const int value =  0; };
// ILS = 6, CHECK_NODE = 31
template <> struct vnode_shift<2, 6, 31,  0> { static const int value =  115; };
template <> struct vnode_shift<2, 6, 31,  1> { static const int value =  189; };
template <> struct vnode_shift<2, 6, 31,  2> { static const int value =  0; };
// ILS = 6, CHECK_NODE = 32
template <> struct vnode_shift<2, 6, 32,  0> { static const int value =  110; };
template <> struct vnode_shift<2, 6, 32,  1> { static const int value =  0; };
template <> struct vnode_shift<2, 6, 32,  2> { static const int value =  163; };
template <> struct vnode_shift<2, 6, 32,  3> { static const int value =  0; };
// ILS = 6, CHECK_NODE = 33
template <> struct vnode_shift<2, 6, 33,  0> { static const int value =  163; };
template <> struct vnode_shift<2, 6, 33,  1> { static const int value =  173; };
template <> struct vnode_shift<2, 6, 33,  2> { static const int value =  179; };
template <> struct vnode_shift<2, 6, 33,  3> { static const int value =  0; };
// ILS = 6, CHECK_NODE = 34
template <> struct vnode_shift<2, 6, 34,  0> { static const int value =  197; };
template <> struct vnode_shift<2, 6, 34,  1> { static const int value =  191; };
template <> struct vnode_shift<2, 6, 34,  2> { static const int value =  193; };
template <> struct vnode_shift<2, 6, 34,  3> { static const int value =  0; };
// ILS = 6, CHECK_NODE = 35
template <> struct vnode_shift<2, 6, 35,  0> { static const int value =  157; };
template <> struct vnode_shift<2, 6, 35,  1> { static const int value =  167; };
template <> struct vnode_shift<2, 6, 35,  2> { static const int value =  181; };
template <> struct vnode_shift<2, 6, 35,  3> { static const int value =  0; };
// ILS = 6, CHECK_NODE = 36
template <> struct vnode_shift<2, 6, 36,  0> { static const int value =  197; };
template <> struct vnode_shift<2, 6, 36,  1> { static const int value =  167; };
template <> struct vnode_shift<2, 6, 36,  2> { static const int value =  179; };
template <> struct vnode_shift<2, 6, 36,  3> { static const int value =  0; };
// ILS = 6, CHECK_NODE = 37
template <> struct vnode_shift<2, 6, 37,  0> { static const int value =  181; };
template <> struct vnode_shift<2, 6, 37,  1> { static const int value =  193; };
template <> struct vnode_shift<2, 6, 37,  2> { static const int value =  0; };
// ILS = 6, CHECK_NODE = 38
template <> struct vnode_shift<2, 6, 38,  0> { static const int value =  157; };
template <> struct vnode_shift<2, 6, 38,  1> { static const int value =  173; };
template <> struct vnode_shift<2, 6, 38,  2> { static const int value =  191; };
template <> struct vnode_shift<2, 6, 38,  3> { static const int value =  0; };
// ILS = 6, CHECK_NODE = 39
template <> struct vnode_shift<2, 6, 39,  0> { static const int value =  181; };
template <> struct vnode_shift<2, 6, 39,  1> { static const int value =  157; };
template <> struct vnode_shift<2, 6, 39,  2> { static const int value =  173; };
template <> struct vnode_shift<2, 6, 39,  3> { static const int value =  0; };
// ILS = 6, CHECK_NODE = 40
template <> struct vnode_shift<2, 6, 40,  0> { static const int value =  193; };
template <> struct vnode_shift<2, 6, 40,  1> { static const int value =  163; };
template <> struct vnode_shift<2, 6, 40,  2> { static const int value =  179; };
template <> struct vnode_shift<2, 6, 40,  3> { static const int value =  0; };
// ILS = 6, CHECK_NODE = 41
template <> struct vnode_shift<2, 6, 41,  0> { static const int value =  191; };
template <> struct vnode_shift<2, 6, 41,  1> { static const int value =  197; };
template <> struct vnode_shift<2, 6, 41,  2> { static const int value =  167; };
template <> struct vnode_shift<2, 6, 41,  3> { static const int value =  0; };
// ILS = 7, CHECK_NODE = 0
template <> struct vnode_shift<2, 7, 0,  0> { static const int value =  145; };
template <> struct vnode_shift<2, 7, 0,  1> { static const int value =  131; };
template <> struct vnode_shift<2, 7, 0,  2> { static const int value =  71; };
template <> struct vnode_shift<2, 7, 0,  3> { static const int value =  21; };
template <> struct vnode_shift<2, 7, 0,  4> { static const int value =  23; };
template <> struct vnode_shift<2, 7, 0,  5> { static const int value =  112; };
template <> struct vnode_shift<2, 7, 0,  6> { static const int value =  1; };
template <> struct vnode_shift<2, 7, 0,  7> { static const int value =  0; };
// ILS = 7, CHECK_NODE = 1
template <> struct vnode_shift<2, 7, 1,  0> { static const int value =  142; };
template <> struct vnode_shift<2, 7, 1,  1> { static const int value =  174; };
template <> struct vnode_shift<2, 7, 1,  2> { static const int value =  183; };
template <> struct vnode_shift<2, 7, 1,  3> { static const int value =  27; };
template <> struct vnode_shift<2, 7, 1,  4> { static const int value =  96; };
template <> struct vnode_shift<2, 7, 1,  5> { static const int value =  23; };
template <> struct vnode_shift<2, 7, 1,  6> { static const int value =  9; };
template <> struct vnode_shift<2, 7, 1,  7> { static const int value =  167; };
template <> struct vnode_shift<2, 7, 1,  8> { static const int value =  0; };
template <> struct vnode_shift<2, 7, 1,  9> { static const int value =  0; };
// ILS = 7, CHECK_NODE = 2
template <> struct vnode_shift<2, 7, 2,  0> { static const int value =  74; };
template <> struct vnode_shift<2, 7, 2,  1> { static const int value =  31; };
template <> struct vnode_shift<2, 7, 2,  2> { static const int value =  3; };
template <> struct vnode_shift<2, 7, 2,  3> { static const int value =  53; };
template <> struct vnode_shift<2, 7, 2,  4> { static const int value =  155; };
template <> struct vnode_shift<2, 7, 2,  5> { static const int value =  0; };
template <> struct vnode_shift<2, 7, 2,  6> { static const int value =  0; };
template <> struct vnode_shift<2, 7, 2,  7> { static const int value =  0; };
// ILS = 7, CHECK_NODE = 3
template <> struct vnode_shift<2, 7, 3,  0> { static const int value =  239; };
template <> struct vnode_shift<2, 7, 3,  1> { static const int value =  171; };
template <> struct vnode_shift<2, 7, 3,  2> { static const int value =  95; };
template <> struct vnode_shift<2, 7, 3,  3> { static const int value =  110; };
template <> struct vnode_shift<2, 7, 3,  4> { static const int value =  159; };
template <> struct vnode_shift<2, 7, 3,  5> { static const int value =  199; };
template <> struct vnode_shift<2, 7, 3,  6> { static const int value =  43; };
template <> struct vnode_shift<2, 7, 3,  7> { static const int value =  75; };
template <> struct vnode_shift<2, 7, 3,  8> { static const int value =  1; };
template <> struct vnode_shift<2, 7, 3,  9> { static const int value =  0; };
// ILS = 7, CHECK_NODE = 4
template <> struct vnode_shift<2, 7, 4,  0> { static const int value =  29; };
template <> struct vnode_shift<2, 7, 4,  1> { static const int value =  140; };
template <> struct vnode_shift<2, 7, 4,  2> { static const int value =  180; };
template <> struct vnode_shift<2, 7, 4,  3> { static const int value =  0; };
// ILS = 7, CHECK_NODE = 5
template <> struct vnode_shift<2, 7, 5,  0> { static const int value =  121; };
template <> struct vnode_shift<2, 7, 5,  1> { static const int value =  41; };
template <> struct vnode_shift<2, 7, 5,  2> { static const int value =  169; };
template <> struct vnode_shift<2, 7, 5,  3> { static const int value =  88; };
template <> struct vnode_shift<2, 7, 5,  4> { static const int value =  207; };
template <> struct vnode_shift<2, 7, 5,  5> { static const int value =  0; };
// ILS = 7, CHECK_NODE = 6
template <> struct vnode_shift<2, 7, 6,  0> { static const int value =  137; };
template <> struct vnode_shift<2, 7, 6,  1> { static const int value =  72; };
template <> struct vnode_shift<2, 7, 6,  2> { static const int value =  172; };
template <> struct vnode_shift<2, 7, 6,  3> { static const int value =  124; };
template <> struct vnode_shift<2, 7, 6,  4> { static const int value =  56; };
template <> struct vnode_shift<2, 7, 6,  5> { static const int value =  0; };
// ILS = 7, CHECK_NODE = 7
template <> struct vnode_shift<2, 7, 7,  0> { static const int value =  86; };
template <> struct vnode_shift<2, 7, 7,  1> { static const int value =  186; };
template <> struct vnode_shift<2, 7, 7,  2> { static const int value =  87; };
template <> struct vnode_shift<2, 7, 7,  3> { static const int value =  172; };
template <> struct vnode_shift<2, 7, 7,  4> { static const int value =  154; };
template <> struct vnode_shift<2, 7, 7,  5> { static const int value =  0; };
// ILS = 7, CHECK_NODE = 8
template <> struct vnode_shift<2, 7, 8,  0> { static const int value =  176; };
template <> struct vnode_shift<2, 7, 8,  1> { static const int value =  169; };
template <> struct vnode_shift<2, 7, 8,  2> { static const int value =  225; };
template <> struct vnode_shift<2, 7, 8,  3> { static const int value =  0; };
// ILS = 7, CHECK_NODE = 9
template <> struct vnode_shift<2, 7, 9,  0> { static const int value =  167; };
template <> struct vnode_shift<2, 7, 9,  1> { static const int value =  238; };
template <> struct vnode_shift<2, 7, 9,  2> { static const int value =  48; };
template <> struct vnode_shift<2, 7, 9,  3> { static const int value =  68; };
template <> struct vnode_shift<2, 7, 9,  4> { static const int value =  0; };
// ILS = 7, CHECK_NODE = 10
template <> struct vnode_shift<2, 7, 10,  0> { static const int value =  38; };
template <> struct vnode_shift<2, 7, 10,  1> { static const int value =  217; };
template <> struct vnode_shift<2, 7, 10,  2> { static const int value =  208; };
template <> struct vnode_shift<2, 7, 10,  3> { static const int value =  232; };
template <> struct vnode_shift<2, 7, 10,  4> { static const int value =  0; };
// ILS = 7, CHECK_NODE = 11
template <> struct vnode_shift<2, 7, 11,  0> { static const int value =  178; };
template <> struct vnode_shift<2, 7, 11,  1> { static const int value =  214; };
template <> struct vnode_shift<2, 7, 11,  2> { static const int value =  168; };
template <> struct vnode_shift<2, 7, 11,  3> { static const int value =  51; };
template <> struct vnode_shift<2, 7, 11,  4> { static const int value =  0; };
// ILS = 7, CHECK_NODE = 12
template <> struct vnode_shift<2, 7, 12,  0> { static const int value =  124; };
template <> struct vnode_shift<2, 7, 12,  1> { static const int value =  122; };
template <> struct vnode_shift<2, 7, 12,  2> { static const int value =  72; };
template <> struct vnode_shift<2, 7, 12,  3> { static const int value =  0; };
// ILS = 7, CHECK_NODE = 13
template <> struct vnode_shift<2, 7, 13,  0> { static const int value =  48; };
template <> struct vnode_shift<2, 7, 13,  1> { static const int value =  57; };
template <> struct vnode_shift<2, 7, 13,  2> { static const int value =  167; };
template <> struct vnode_shift<2, 7, 13,  3> { static const int value =  219; };
template <> struct vnode_shift<2, 7, 13,  4> { static const int value =  0; };
// ILS = 7, CHECK_NODE = 14
template <> struct vnode_shift<2, 7, 14,  0> { static const int value =  82; };
template <> struct vnode_shift<2, 7, 14,  1> { static const int value =  232; };
template <> struct vnode_shift<2, 7, 14,  2> { static const int value =  204; };
template <> struct vnode_shift<2, 7, 14,  3> { static const int value =  162; };
template <> struct vnode_shift<2, 7, 14,  4> { static const int value =  0; };
// ILS = 7, CHECK_NODE = 15
template <> struct vnode_shift<2, 7, 15,  0> { static const int value =  38; };
template <> struct vnode_shift<2, 7, 15,  1> { static const int value =  217; };
template <> struct vnode_shift<2, 7, 15,  2> { static const int value =  157; };
template <> struct vnode_shift<2, 7, 15,  3> { static const int value =  0; };
// ILS = 7, CHECK_NODE = 16
template <> struct vnode_shift<2, 7, 16,  0> { static const int value =  170; };
template <> struct vnode_shift<2, 7, 16,  1> { static const int value =  23; };
template <> struct vnode_shift<2, 7, 16,  2> { static const int value =  175; };
template <> struct vnode_shift<2, 7, 16,  3> { static const int value =  202; };
template <> struct vnode_shift<2, 7, 16,  4> { static const int value =  0; };
// ILS = 7, CHECK_NODE = 17
template <> struct vnode_shift<2, 7, 17,  0> { static const int value =  196; };
template <> struct vnode_shift<2, 7, 17,  1> { static const int value =  173; };
template <> struct vnode_shift<2, 7, 17,  2> { static const int value =  195; };
template <> struct vnode_shift<2, 7, 17,  3> { static const int value =  218; };
template <> struct vnode_shift<2, 7, 17,  4> { static const int value =  0; };
// ILS = 7, CHECK_NODE = 18
template <> struct vnode_shift<2, 7, 18,  0> { static const int value =  128; };
template <> struct vnode_shift<2, 7, 18,  1> { static const int value =  211; };
template <> struct vnode_shift<2, 7, 18,  2> { static const int value =  210; };
template <> struct vnode_shift<2, 7, 18,  3> { static const int value =  0; };
// ILS = 7, CHECK_NODE = 19
template <> struct vnode_shift<2, 7, 19,  0> { static const int value =  39; };
template <> struct vnode_shift<2, 7, 19,  1> { static const int value =  84; };
template <> struct vnode_shift<2, 7, 19,  2> { static const int value =  88; };
template <> struct vnode_shift<2, 7, 19,  3> { static const int value =  0; };
// ILS = 7, CHECK_NODE = 20
template <> struct vnode_shift<2, 7, 20,  0> { static const int value =  117; };
template <> struct vnode_shift<2, 7, 20,  1> { static const int value =  227; };
template <> struct vnode_shift<2, 7, 20,  2> { static const int value =  6; };
template <> struct vnode_shift<2, 7, 20,  3> { static const int value =  0; };
// ILS = 7, CHECK_NODE = 21
template <> struct vnode_shift<2, 7, 21,  0> { static const int value =  238; };
template <> struct vnode_shift<2, 7, 21,  1> { static const int value =  13; };
template <> struct vnode_shift<2, 7, 21,  2> { static const int value =  11; };
template <> struct vnode_shift<2, 7, 21,  3> { static const int value =  0; };
// ILS = 7, CHECK_NODE = 22
template <> struct vnode_shift<2, 7, 22,  0> { static const int value =  195; };
template <> struct vnode_shift<2, 7, 22,  1> { static const int value =  44; };
template <> struct vnode_shift<2, 7, 22,  2> { static const int value =  0; };
// ILS = 7, CHECK_NODE = 23
template <> struct vnode_shift<2, 7, 23,  0> { static const int value =  5; };
template <> struct vnode_shift<2, 7, 23,  1> { static const int value =  94; };
template <> struct vnode_shift<2, 7, 23,  2> { static const int value =  111; };
template <> struct vnode_shift<2, 7, 23,  3> { static const int value =  0; };
// ILS = 7, CHECK_NODE = 24
template <> struct vnode_shift<2, 7, 24,  0> { static const int value =  81; };
template <> struct vnode_shift<2, 7, 24,  1> { static const int value =  19; };
template <> struct vnode_shift<2, 7, 24,  2> { static const int value =  130; };
template <> struct vnode_shift<2, 7, 24,  3> { static const int value =  0; };
// ILS = 7, CHECK_NODE = 25
template <> struct vnode_shift<2, 7, 25,  0> { static const int value =  66; };
template <> struct vnode_shift<2, 7, 25,  1> { static const int value =  95; };
template <> struct vnode_shift<2, 7, 25,  2> { static const int value =  0; };
// ILS = 7, CHECK_NODE = 26
template <> struct vnode_shift<2, 7, 26,  0> { static const int value =  146; };
template <> struct vnode_shift<2, 7, 26,  1> { static const int value =  66; };
template <> struct vnode_shift<2, 7, 26,  2> { static const int value =  190; };
template <> struct vnode_shift<2, 7, 26,  3> { static const int value =  86; };
template <> struct vnode_shift<2, 7, 26,  4> { static const int value =  0; };
// ILS = 7, CHECK_NODE = 27
template <> struct vnode_shift<2, 7, 27,  0> { static const int value =  64; };
template <> struct vnode_shift<2, 7, 27,  1> { static const int value =  181; };
template <> struct vnode_shift<2, 7, 27,  2> { static const int value =  0; };
// ILS = 7, CHECK_NODE = 28
template <> struct vnode_shift<2, 7, 28,  0> { static const int value =  7; };
template <> struct vnode_shift<2, 7, 28,  1> { static const int value =  144; };
template <> struct vnode_shift<2, 7, 28,  2> { static const int value =  16; };
template <> struct vnode_shift<2, 7, 28,  3> { static const int value =  0; };
// ILS = 7, CHECK_NODE = 29
template <> struct vnode_shift<2, 7, 29,  0> { static const int value =  25; };
template <> struct vnode_shift<2, 7, 29,  1> { static const int value =  57; };
template <> struct vnode_shift<2, 7, 29,  2> { static const int value =  0; };
// ILS = 7, CHECK_NODE = 30
template <> struct vnode_shift<2, 7, 30,  0> { static const int value =  37; };
template <> struct vnode_shift<2, 7, 30,  1> { static const int value =  139; };
template <> struct vnode_shift<2, 7, 30,  2> { static const int value =  221; };
template <> struct vnode_shift<2, 7, 30,  3> { static const int value =  17; };
template <> struct vnode_shift<2, 7, 30,  4> { static const int value =  0; };
// ILS = 7, CHECK_NODE = 31
template <> struct vnode_shift<2, 7, 31,  0> { static const int value =  201; };
template <> struct vnode_shift<2, 7, 31,  1> { static const int value =  46; };
template <> struct vnode_shift<2, 7, 31,  2> { static const int value =  0; };
// ILS = 7, CHECK_NODE = 32
template <> struct vnode_shift<2, 7, 32,  0> { static const int value =  179; };
template <> struct vnode_shift<2, 7, 32,  1> { static const int value =  14; };
template <> struct vnode_shift<2, 7, 32,  2> { static const int value =  116; };
template <> struct vnode_shift<2, 7, 32,  3> { static const int value =  0; };
// ILS = 7, CHECK_NODE = 33
template <> struct vnode_shift<2, 7, 33,  0> { static const int value =  46; };
template <> struct vnode_shift<2, 7, 33,  1> { static const int value =  2; };
template <> struct vnode_shift<2, 7, 33,  2> { static const int value =  106; };
template <> struct vnode_shift<2, 7, 33,  3> { static const int value =  0; };
// ILS = 7, CHECK_NODE = 34
template <> struct vnode_shift<2, 7, 34,  0> { static const int value =  184; };
template <> struct vnode_shift<2, 7, 34,  1> { static const int value =  135; };
template <> struct vnode_shift<2, 7, 34,  2> { static const int value =  141; };
template <> struct vnode_shift<2, 7, 34,  3> { static const int value =  0; };
// ILS = 7, CHECK_NODE = 35
template <> struct vnode_shift<2, 7, 35,  0> { static const int value =  85; };
template <> struct vnode_shift<2, 7, 35,  1> { static const int value =  225; };
template <> struct vnode_shift<2, 7, 35,  2> { static const int value =  175; };
template <> struct vnode_shift<2, 7, 35,  3> { static const int value =  0; };
// ILS = 7, CHECK_NODE = 36
template <> struct vnode_shift<2, 7, 36,  0> { static const int value =  178; };
template <> struct vnode_shift<2, 7, 36,  1> { static const int value =  112; };
template <> struct vnode_shift<2, 7, 36,  2> { static const int value =  106; };
template <> struct vnode_shift<2, 7, 36,  3> { static const int value =  0; };
// ILS = 7, CHECK_NODE = 37
template <> struct vnode_shift<2, 7, 37,  0> { static const int value =  154; };
template <> struct vnode_shift<2, 7, 37,  1> { static const int value =  114; };
template <> struct vnode_shift<2, 7, 37,  2> { static const int value =  0; };
// ILS = 7, CHECK_NODE = 38
template <> struct vnode_shift<2, 7, 38,  0> { static const int value =  42; };
template <> struct vnode_shift<2, 7, 38,  1> { static const int value =  41; };
template <> struct vnode_shift<2, 7, 38,  2> { static const int value =  105; };
template <> struct vnode_shift<2, 7, 38,  3> { static const int value =  0; };
// ILS = 7, CHECK_NODE = 39
template <> struct vnode_shift<2, 7, 39,  0> { static const int value =  167; };
template <> struct vnode_shift<2, 7, 39,  1> { static const int value =  45; };
template <> struct vnode_shift<2, 7, 39,  2> { static const int value =  189; };
template <> struct vnode_shift<2, 7, 39,  3> { static const int value =  0; };
// ILS = 7, CHECK_NODE = 40
template <> struct vnode_shift<2, 7, 40,  0> { static const int value =  78; };
template <> struct vnode_shift<2, 7, 40,  1> { static const int value =  67; };
template <> struct vnode_shift<2, 7, 40,  2> { static const int value =  180; };
template <> struct vnode_shift<2, 7, 40,  3> { static const int value =  0; };
// ILS = 7, CHECK_NODE = 41
template <> struct vnode_shift<2, 7, 41,  0> { static const int value =  53; };
template <> struct vnode_shift<2, 7, 41,  1> { static const int value =  215; };
template <> struct vnode_shift<2, 7, 41,  2> { static const int value =  230; };
template <> struct vnode_shift<2, 7, 41,  3> { static const int value =  0; };


////////////////////////////////////////////////////////////////////////
// vnode_shift_mod
// Variable node shift values as a function of base graph, lifting size
// (Z), check node index, and index within the row. The modulo operation
// (on lifting size) is performed.
// See Tables 5.3.2-2 and 5.3.2-3, 3GPP 38.212
template <int BG, int Z, int CHECK_NODE, int INDEX> struct vnode_shift_mod
{
    static const unsigned short value = vnode_shift<BG, set_index<Z>::value, CHECK_NODE, INDEX>::value % Z;
};

////////////////////////////////////////////////////////////////////////
// vnode_base_offset
// Returns the index of the first APP value for the requested non-zero
// block matrix in the CHECK_NODE row.
// value = col_index * Z
template <int BG, int Z, int CHECK_NODE, int INDEX> struct vnode_base_offset
{
    static const unsigned short value = vnode_index<BG, CHECK_NODE, INDEX>::value * Z;
};

////////////////////////////////////////////////////////////////////////
// wrap_index
// --base offset---->|
//     <---shift---->|
//     | - - - - - - - - - - - - - - - - - - - - |
//  0  |             x                           |
//  1  |               x                         |
//  2  |                 x                       |
//  3  |                   x                     |
//  4  |                     x                   |
//  5  |                       x                 |
//  6  |                         x               |
//  7  |                           x             |
//  8  |                             x           |
//  9  |                               x         |
// 10  |                                 x       |
// 11  |                                   x     |
// 12  |                                     x   |
// 13  |                                       x |
// 14  | x.......................................|<-- wrap index
// 15  |   x                                     |
// 16  |     x                                   |
// 17  |       x                                 |
// 18  |         x                               |
// 19  |           x                             |
//     | - - - - - - - - - - - - - - - - - - - - |
template <int BG, int Z, int CHECK_NODE, int INDEX> struct wrap_index
{
    static const unsigned short value = Z - vnode_shift_mod<BG, Z, CHECK_NODE, INDEX>::value;
};

////////////////////////////////////////////////////////////////////////
// vnode_shift_offset
// Provides (vnode_column * Z) + shift_mod for given values of the
// base graph (BG), lifting size (Z), CHECK_NODE, and row index INDEX.
template <int BG, int Z, int CHECK_NODE, int INDEX> struct vnode_shift_offset
{
    static const unsigned short value = (vnode_index<BG, CHECK_NODE, INDEX>::value * Z) +
                                        vnode_shift_mod<BG, Z, CHECK_NODE, INDEX>::value;
};

////////////////////////////////////////////////////////////////////////
// vnode_adj_shift_offset
// Provides [(vnode_column - 1) * Z] + shift_mod for given values of the
// base graph (BG), lifting size (Z), CHECK_NODE, and row index INDEX.
template <int BG, int Z, int CHECK_NODE, int INDEX> struct vnode_adj_shift_offset
{
    static const int32_t value = ((static_cast<int32_t>(vnode_index<BG, CHECK_NODE, INDEX>::value) - 1) * Z) +
                                 static_cast<int32_t>(vnode_shift_mod<BG, Z, CHECK_NODE, INDEX>::value);
};

////////////////////////////////////////////////////////////////////////
// index_out_of_bounds
// Value used to represent an "out of bounds" access - typically
// used only when accessing PAIRS in a row with an odd row degree.
struct index_out_of_bounds
{
    static const unsigned short value = 0;
};

////////////////////////////////////////////////////////////////////////
// u16x2_to_u32_t
template <unsigned short u0, unsigned short u1>
struct u16x2_to_u32_t
{
    static const unsigned int value = u0 + (u1 << 16);
};

////////////////////////////////////////////////////////////////////////
// Return the maximum of two integers
template <int A, int B>
struct max_t
{
    static const int value = (A < B) ? B : A;
};

////////////////////////////////////////////////////////////////////////
// vnode_shift_mod_if
// Variable node shift values as a function of base graph, lifting size
// (Z), check node index, and index within the row. The modulo operation
// (on lifting size) is performed. If the index within the row is
// greater than or equal to the row degree, the index_out_of_bounds
// structure will be used to provide a "dummy" value.
template <int BG, int Z, int CHECK_NODE, int INDEX> struct vnode_shift_mod_if
{
    static const unsigned short value = std::conditional<INDEX < row_degree<BG, CHECK_NODE>::value,
                                                         vnode_shift_mod<BG, Z, CHECK_NODE, INDEX>,
                                                         index_out_of_bounds>::type::value;
};

////////////////////////////////////////////////////////////////////////
// vnode_shift_mod_pair
// Combines two adjacent vnode_shift_mod values into a single word.
template <int BG, int Z, int CHECK_NODE, int PAIR_INDEX> struct vnode_shift_mod_pair
{
    static const unsigned int value = u16x2_to_u32_t<vnode_shift_mod_if<BG, Z, CHECK_NODE, PAIR_INDEX * 2>::value,
                                                     vnode_shift_mod_if<BG, Z, CHECK_NODE, PAIR_INDEX * 2 + 1>::value>::value;
};

////////////////////////////////////////////////////////////////////////
// vnode_base_offset_if
// If the index within the row is
// greater than or equal to the row degree, the index_out_of_bounds
// structure will be used to provide a "dummy" value.
// value = col_index * Z
template <int BG, int Z, int CHECK_NODE, int INDEX> struct vnode_base_offset_if
{
    static const unsigned short value = std::conditional<INDEX < row_degree<BG, CHECK_NODE>::value,
                                                         vnode_base_offset<BG, Z, CHECK_NODE, INDEX>,
                                                         index_out_of_bounds>::type::value;
};

////////////////////////////////////////////////////////////////////////
// vnode_base_offset_pair
// Combines two adjacent vnode_shift_mod values into a single word.
template <int BG, int Z, int CHECK_NODE, int PAIR_INDEX> struct vnode_base_offset_pair
{
    static const unsigned int value = u16x2_to_u32_t<vnode_base_offset_if<BG, Z, CHECK_NODE, PAIR_INDEX * 2>::value,
                                                     vnode_base_offset_if<BG, Z, CHECK_NODE, PAIR_INDEX * 2 + 1>::value>::value;
};

////////////////////////////////////////////////////////////////////////
// vnode_index_if
// Variable node index as a function of base graph, check node index,
// and index within the row. If the index within the row is greater than
// or equal to the row degree, the index_out_of_bounds
// structure will be used to provide a "dummy" value.
template <int BG, int CHECK_NODE, int INDEX> struct vnode_index_if
{
    static const unsigned short value = std::conditional<INDEX < row_degree<BG, CHECK_NODE>::value,
                                                         vnode_index<BG, CHECK_NODE, INDEX>,
                                                         index_out_of_bounds>::type::value;
};

////////////////////////////////////////////////////////////////////////
// wrap_index_if
// Wrap index as a function of base graph, check node index, and index
// within the row. If the index within the row is greater than or equal
// to the row degree, the index_out_of_bounds structure will be used to
// provide a "dummy" value.
template <int BG, int Z, int CHECK_NODE, int INDEX> struct wrap_index_if
{
    static const unsigned short value = std::conditional<INDEX < row_degree<BG, CHECK_NODE>::value,
                                                         wrap_index<BG, Z, CHECK_NODE, INDEX>,
                                                         index_out_of_bounds>::type::value;
};

////////////////////////////////////////////////////////////////////////
// wrap_index_pair
template <int BG, int Z, int CHECK_NODE, int PAIR_INDEX> struct wrap_index_pair
{
    static const unsigned int value = u16x2_to_u32_t<wrap_index_if<BG, Z, CHECK_NODE, PAIR_INDEX * 2>::value,
                                                     wrap_index_if<BG, Z, CHECK_NODE, PAIR_INDEX * 2 + 1>::value>::value;
};

////////////////////////////////////////////////////////////////////////
// vnode_shift_offset_if
// Provides (vnode_colum * Z) + shift_mod for given values of the
// base graph (BG), lifting size (Z), CHECK_NODE, and row index INDEX.
template <int BG, int Z, int CHECK_NODE, int INDEX> struct vnode_shift_offset_if
{
    static const unsigned short value = std::conditional<INDEX < row_degree<BG, CHECK_NODE>::value,
                                                         vnode_shift_offset<BG, Z, CHECK_NODE, INDEX>,
                                                         index_out_of_bounds>::type::value;
};

////////////////////////////////////////////////////////////////////////
// vnode_shift_offset_pair
// NOTE that the calculated values may be too large to use with
// denormalized floats.
template <int BG, int Z, int CHECK_NODE, int PAIR_INDEX> struct vnode_shift_offset_pair
{
    static const unsigned int value = u16x2_to_u32_t<vnode_shift_offset_if<BG, Z, CHECK_NODE, PAIR_INDEX * 2>::value,
                                                     vnode_shift_offset_if<BG, Z, CHECK_NODE, PAIR_INDEX * 2 + 1>::value>::value;
};

////////////////////////////////////////////////////////////////////////
// vnode_adj_shift_offset_if
// Provides ((vnode_colum-1) * Z) + shift_mod for given values of the
// base graph (BG), lifting size (Z), CHECK_NODE, and row index INDEX.
template <int BG, int Z, int CHECK_NODE, int INDEX> struct vnode_adj_shift_offset_if
{
    static const int32_t value = std::conditional<INDEX < row_degree<BG, CHECK_NODE>::value,
                                                  vnode_adj_shift_offset<BG, Z, CHECK_NODE, INDEX>,
                                                  index_out_of_bounds>::type::value;
};

////////////////////////////////////////////////////////////////////////
// max_app_address
// Template struct with a value constant that returns the maximum app
// address for a pair of nonzero row variables. (This is useful when
// using x2 SIMD instructions to perform address calculations.)
template <typename T, int BG, int Z, int CHECK_NODE, int COL_INDEX> struct max_app_address
{
    static const int value = (vnode_base_offset_if<BG, Z, CHECK_NODE, COL_INDEX>::value + (Z - 1)) * sizeof(T);
};

////////////////////////////////////////////////////////////////////////
// max_pair_app_address
template <typename T, int BG, int Z, int CHECK_NODE, int PAIR_INDEX> struct max_pair_app_address
{
    static const int value = max_t<max_app_address<T, BG, Z, CHECK_NODE, PAIR_INDEX * 2>    ::value,
                                   max_app_address<T, BG, Z, CHECK_NODE, PAIR_INDEX * 2 + 1>::value>::value;
};
} // namespace ldpc2
#endif // ifndef(NRLDPC_TEMPLATES_CUH_INCLUDED_)
