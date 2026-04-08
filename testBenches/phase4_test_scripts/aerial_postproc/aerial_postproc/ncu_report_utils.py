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

import ncu_report
import pandas as pd

METRIC_LATENCY = 'gpu__time_duration.sum'
METRIC_INST_EXECUTED_FFMA_PEAK = 'sm__sass_thread_inst_executed_op_ffma_pred_on.sum.peak_sustained'
METRIC_INST_EXECUTED_DFMA_PEAK = 'sm__sass_thread_inst_executed_op_dfma_pred_on.sum.peak_sustained'
METRIC_INST_EXECUTED_FADD = 'smsp__sass_thread_inst_executed_op_fadd_pred_on.sum.per_cycle_elapsed'
METRIC_INST_EXECUTED_FMUL = 'smsp__sass_thread_inst_executed_op_fmul_pred_on.sum.per_cycle_elapsed'
METRIC_INST_EXECUTED_FFMA = 'smsp__sass_thread_inst_executed_op_ffma_pred_on.sum.per_cycle_elapsed'
METRIC_INST_EXECUTED_DADD = 'smsp__sass_thread_inst_executed_op_dadd_pred_on.sum.per_cycle_elapsed'
METRIC_INST_EXECUTED_DMUL = 'smsp__sass_thread_inst_executed_op_dmul_pred_on.sum.per_cycle_elapsed'
METRIC_INST_EXECUTED_DFMA = 'smsp__sass_thread_inst_executed_op_dfma_pred_on.sum.per_cycle_elapsed'
METRIC_CYCLES_PER_SECOND  = 'smsp__cycles_elapsed.avg.per_second'
METRIC_SM_COMPUTE_THR     = 'sm__throughput.avg.pct_of_peak_sustained_elapsed'
METRIC_MEMORY_THR         = 'gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed'
METRIC_L1_TEX_CACHE_THR   = 'l1tex__throughput.avg.pct_of_peak_sustained_active'
METRIC_L2_CACHE_THR       = 'lts__throughput.avg.pct_of_peak_sustained_elapsed'
METRIC_DRAM_THR           = 'gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed'
METRIC_THEOR_OCC          = 'sm__maximum_warps_per_active_cycle_pct'
METRIC_THEOR_ACT_WARP_SM  = 'sm__maximum_warps_avg_per_active_cycle'
METRIC_ACHIEVED_OCC       = 'sm__warps_active.avg.pct_of_peak_sustained_active'
METRIC_ACHIEVED_ACT_WARP_SM = 'sm__warps_active.avg.per_cycle_active'
METRIC_BLK_LIM_REGS       = 'launch__occupancy_limit_registers'
METRIC_BLK_LIM_SH_MEM     = 'launch__occupancy_limit_shared_mem'
METRIC_BLK_LIM_WARPS      = 'launch__occupancy_limit_warps'
METRIC_BLK_LIM_SM         = 'launch__occupancy_limit_blocks'
METRIV_BLK_SIZE           = 'launch__block_size'
METRIC_GRID_SIZE          = 'launch__grid_size'
METRIC_SM_COUNT           = 'launch__sm_count'
METRIC_USES_GC            = 'launch__uses_green_context'
METRIC_WARP_CYCLES_PER_ISSUED_INTST = 'smsp__average_warp_latency_per_inst_issued.ratio'
METRIC_AVG_ACT_THREADS_PER_WARP     = 'smsp__thread_inst_executed_per_inst_executed.ratio'
METRIC_WARP_CYCLES_PER_EXEC_INST    = 'smsp__average_warps_active_per_inst_executed.ratio'
METRIC_AVG_NOT_PRED_THREADS_PER_WARP = 'smsp__thread_inst_executed_pred_on_per_inst_executed.ratio'
METRIC_STALL_DRAIN    = 'smsp__average_warps_issue_stalled_drain_per_issue_active.ratio'
METRIC_STALL_IMC_MISS = 'smsp__average_warps_issue_stalled_imc_miss_per_issue_active.ratio'
METRIC_STALL_BARRIER  = 'smsp__average_warps_issue_stalled_barrier_per_issue_active.ratio'
# METRIC_STALL_GMMA     = 'smsp__average_warps_issue_stalled_gmma_per_issue_active.ratio'
METRIC_STALL_BRANCH_RESOLVING = 'smsp__average_warps_issue_stalled_branch_resolving_per_issue_active.ratio'
METRIC_STALL_MEMBAR   = 'smsp__average_warps_issue_stalled_membar_per_issue_active.ratio'
METRIC_STALL_SHORT_SCOREBOARD = 'smsp__average_warps_issue_stalled_short_scoreboard_per_issue_active.ratio'
METRIC_STALL_SLEEPING = 'smsp__average_warps_issue_stalled_sleeping_per_issue_active.ratio'
METRIC_STALL_WAIT     = 'smsp__average_warps_issue_stalled_wait_per_issue_active.ratio'
METRIC_STALL_NO_INSTR = 'smsp__average_warps_issue_stalled_no_instruction_per_issue_active.ratio'
METRIC_STALL_MATH_PIPE_THROTTLE = 'smsp__average_warps_issue_stalled_math_pipe_throttle_per_issue_active.ratio'
METRIC_STALL_TEXT_THROTTLE      = 'smsp__average_warps_issue_stalled_tex_throttle_per_issue_active.ratio'
METRIC_STALL_LG_THROTTLE        = 'smsp__average_warps_issue_stalled_lg_throttle_per_issue_active.ratio'
METRIC_STALL_DISPATH_STALL      = 'smsp__average_warps_issue_stalled_dispatch_stall_per_issue_active.ratio'
METRIC_STALL_MISC               = 'smsp__average_warps_issue_stalled_misc_per_issue_active.ratio'
METRIC_STALL_NOT_SELECTED       = 'smsp__average_warps_issue_stalled_not_selected_per_issue_active.ratio'
METRIC_STALL_SELECTED           = 'smsp__average_warps_issue_stalled_selected_per_issue_active.ratio'
METRIC_STALL_LONG_SCOREBOARD    = 'smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio'
METRIC_STALL_MIO_THROTTLE       = 'smsp__average_warps_issue_stalled_mio_throttle_per_issue_active.ratio'

def get_kernel_metrics(report_filename: str) -> pd.DataFrame:
    report = ncu_report.load_report(report_filename)
    num_ranges = report.num_ranges()
    if num_ranges == 0:
        return pd.empty()
    my_range = report.range_by_idx(0)

    kernel_names = []
    meas_latency = []
    meas_single_precision_gflops = []
    meas_single_precision_to_peak_pct = []
    meas_double_precision_gflops = []
    meas_double_precision_to_peak_pct = []
    meas_compute_thr = []
    meas_memory_thr  = []
    meas_l1_tex_cache_thr = []
    meas_l2_cache_thr     = []
    meas_dram_thr         = []
    meas_theor_occ        = []
    meas_theor_act_warp_sm = []
    meas_achieved_occ        = []
    meas_achieved_act_warp_sm = []
    meas_blk_lim_regs = []
    meas_blk_lim_sh_mem = []
    meas_blk_lim_warps = []
    meas_blk_lim_sm = []
    meas_grid_size_sm = []
    meas_blk_size = []
    meas_sm_count = []
    meas_uses_gc = []
    meas_warp_cycles_per_issued_inst = []
    meas_avg_act_threads_per_warp = []
    meas_avg_not_pred_threads_per_warp = []
    meas_warp_cycles_per_Exec_inst = []
    meas_stall_drain = []
    meas_stall_imc_miss = []
    meas_stall_barrier = []
    # meas_stall_gmma = []
    meas_stall_branch_resolving = []
    meas_stall_membar = []
    meas_stall_short_scoreboard = [] 
    meas_stall_sleeping = []
    meas_stall_wait = []
    meas_stall_no_instr = []
    meas_stall_math_pipe_throttle = []
    meas_stall_text_throttle = []
    meas_stall_lg_throttle = []
    meas_stall_dispath_stall = []
    meas_stall_misc = []
    meas_stall_not_selected = []
    meas_stall_selected = []
    meas_stall_long_scoreboard = []
    meas_stall_mio_throttle = []   

    for i in range(my_range.num_actions()):
        kernel = my_range.action_by_idx(i)
        
        kernel_names.append(kernel.name())
        meas_latency.append(kernel.metric_by_name(METRIC_LATENCY).value())
        
        cycles_per_second = kernel.metric_by_name(METRIC_CYCLES_PER_SECOND).value() * 1.0 
        inst_executed_single_precision = (kernel.metric_by_name(METRIC_INST_EXECUTED_FADD).value() \
                    + kernel.metric_by_name(METRIC_INST_EXECUTED_FMUL).value() \
                    + 2 * kernel.metric_by_name(METRIC_INST_EXECUTED_FFMA).value()) * cycles_per_second / 1e9
        inst_executed_double_precision = (kernel.metric_by_name(METRIC_INST_EXECUTED_DADD).value() \
                    + kernel.metric_by_name(METRIC_INST_EXECUTED_DMUL).value() \
                    + 2 * kernel.metric_by_name(METRIC_INST_EXECUTED_DFMA).value()) * cycles_per_second / 1e9
        single_precision_peak = kernel.metric_by_name(METRIC_INST_EXECUTED_FFMA_PEAK).value()
        double_precision_peak = kernel.metric_by_name(METRIC_INST_EXECUTED_DFMA_PEAK).value()

        meas_single_precision_gflops.append(inst_executed_single_precision)
        meas_single_precision_to_peak_pct.append(inst_executed_single_precision * 100.0 / single_precision_peak)
        meas_double_precision_gflops.append(inst_executed_double_precision)
        meas_double_precision_to_peak_pct.append(inst_executed_double_precision * 100.0 / double_precision_peak)
        meas_compute_thr.append(kernel.metric_by_name(METRIC_SM_COMPUTE_THR).value())
        meas_memory_thr.append(kernel.metric_by_name(METRIC_MEMORY_THR).value())
        meas_l1_tex_cache_thr.append(kernel.metric_by_name(METRIC_BLK_LIM_WARPS).value())
        meas_l2_cache_thr.append(kernel.metric_by_name(METRIC_L2_CACHE_THR).value())
        meas_dram_thr.append(kernel.metric_by_name(METRIC_DRAM_THR).value())
        meas_theor_occ.append(kernel.metric_by_name(METRIC_THEOR_OCC).value())
        meas_theor_act_warp_sm.append(kernel.metric_by_name(METRIC_THEOR_ACT_WARP_SM).value())
        meas_achieved_occ.append(kernel.metric_by_name(METRIC_ACHIEVED_OCC).value())
        meas_achieved_act_warp_sm.append(kernel.metric_by_name(METRIC_ACHIEVED_ACT_WARP_SM).value())
        meas_blk_lim_regs.append(kernel.metric_by_name(METRIC_BLK_LIM_REGS).value())
        meas_blk_lim_sh_mem.append(kernel.metric_by_name(METRIC_BLK_LIM_SH_MEM).value())
        meas_blk_lim_warps.append(kernel.metric_by_name(METRIC_ACHIEVED_ACT_WARP_SM).value())
        meas_blk_lim_sm.append(kernel.metric_by_name(METRIC_BLK_LIM_SM).value())
        meas_grid_size_sm.append(kernel.metric_by_name(METRIC_GRID_SIZE).value())
        meas_blk_size.append(kernel.metric_by_name(METRIV_BLK_SIZE).value())
        meas_sm_count.append(kernel.metric_by_name(METRIC_SM_COUNT).value())
        meas_uses_gc.append(kernel.metric_by_name(METRIC_USES_GC).value())
        meas_warp_cycles_per_issued_inst.append(kernel.metric_by_name(METRIC_WARP_CYCLES_PER_ISSUED_INTST).value())
        meas_avg_act_threads_per_warp.append(kernel.metric_by_name(METRIC_AVG_ACT_THREADS_PER_WARP).value())
        meas_avg_not_pred_threads_per_warp.append(kernel.metric_by_name(METRIC_WARP_CYCLES_PER_EXEC_INST).value())
        meas_warp_cycles_per_Exec_inst.append(kernel.metric_by_name(METRIC_AVG_NOT_PRED_THREADS_PER_WARP).value())
        meas_stall_drain.append(kernel.metric_by_name(METRIC_STALL_DRAIN).value())
        meas_stall_imc_miss.append(kernel.metric_by_name(METRIC_STALL_IMC_MISS).value())
        meas_stall_barrier.append(kernel.metric_by_name(METRIC_STALL_BARRIER).value())
        # meas_stall_gmma.append(kernel.metric_by_name(METRIC_STALL_GMMA).value())
        meas_stall_branch_resolving.append(kernel.metric_by_name(METRIC_STALL_BRANCH_RESOLVING).value())
        meas_stall_membar.append(kernel.metric_by_name(METRIC_STALL_MEMBAR).value())
        meas_stall_short_scoreboard.append(kernel.metric_by_name(METRIC_STALL_SHORT_SCOREBOARD).value()) 
        meas_stall_sleeping.append(kernel.metric_by_name(METRIC_STALL_SLEEPING).value())
        meas_stall_wait.append(kernel.metric_by_name(METRIC_STALL_WAIT).value())
        meas_stall_no_instr.append(kernel.metric_by_name(METRIC_STALL_NO_INSTR).value())
        meas_stall_math_pipe_throttle.append(kernel.metric_by_name(METRIC_STALL_MATH_PIPE_THROTTLE).value())
        meas_stall_text_throttle.append(kernel.metric_by_name(METRIC_STALL_TEXT_THROTTLE).value())
        meas_stall_lg_throttle.append(kernel.metric_by_name(METRIC_STALL_LG_THROTTLE).value())
        meas_stall_dispath_stall.append(kernel.metric_by_name(METRIC_STALL_DISPATH_STALL).value())
        meas_stall_misc.append(kernel.metric_by_name(METRIC_STALL_MISC).value())
        meas_stall_not_selected.append(kernel.metric_by_name(METRIC_STALL_NOT_SELECTED).value())
        meas_stall_selected.append(kernel.metric_by_name(METRIC_STALL_SELECTED).value())
        meas_stall_long_scoreboard.append(kernel.metric_by_name(METRIC_STALL_LONG_SCOREBOARD).value())
        meas_stall_mio_throttle.append(kernel.metric_by_name(METRIC_STALL_MIO_THROTTLE).value())

    df = pd.DataFrame({
        'kernel': kernel_names,
        'latency': meas_latency,
        'sp_gflops': meas_single_precision_gflops,
        'sp_to_peak_pct': meas_single_precision_to_peak_pct,
        'dp_gflops': meas_double_precision_gflops,
        'dp_to_peak_pct': meas_double_precision_to_peak_pct,
        'compute_thr': meas_compute_thr,
        'memory_thr': meas_memory_thr,
        'l1_tex_cache_thr': meas_l1_tex_cache_thr,
        'l2_cache_thr': meas_l2_cache_thr,
        'dram_thr': meas_dram_thr,
        'occ_theor': meas_theor_occ,
        'act_warp_sm_theor': meas_theor_act_warp_sm,
        'occ_achieved': meas_achieved_occ,
        'act_warp_sm_achieved': meas_achieved_act_warp_sm,
        'blk_lim_regs': meas_blk_lim_regs,
        'blk_lim_sh_mem': meas_blk_lim_sh_mem,
        'blk_lim_warps': meas_blk_lim_warps,
        'blk_lim_sm': meas_blk_lim_sm,
        'grid_size': meas_grid_size_sm,
        'blk_size': meas_blk_size,
        'sm_count': meas_sm_count,
        'gc': meas_uses_gc,
        'warp_cycles_per_issued_inst' : meas_warp_cycles_per_issued_inst,
        'avg_act_threads_per_warp' : meas_avg_act_threads_per_warp,
        'avg_not_pred_threads_per_warp' : meas_avg_not_pred_threads_per_warp,
        'warp_cycles_per_Exec_inst' : meas_warp_cycles_per_Exec_inst,
        'stall_drain' : meas_stall_drain,
        'stall_imc_miss' : meas_stall_imc_miss,
        'stall_barrier' : meas_stall_barrier,
        # 'stall_gmma' : meas_stall_gmma,
        'stall_branch_resolving' : meas_stall_branch_resolving,
        'stall_membar' : meas_stall_membar,
        'stall_short_scoreboard'  : meas_stall_short_scoreboard, 
        'stall_sleeping' : meas_stall_sleeping,
        'stall_wait' : meas_stall_wait,
        'stall_no_instr' : meas_stall_no_instr,
        'stall_math_pipe_throttle' : meas_stall_math_pipe_throttle,
        'stall_text_throttle' : meas_stall_text_throttle,
        'stall_lg_throttle' : meas_stall_lg_throttle,
        'stall_dispath_stall' : meas_stall_dispath_stall,
        'stall_misc' : meas_stall_misc,
        'stall_not_selected' : meas_stall_not_selected,
        'stall_selected' : meas_stall_selected,
        'stall_long_scoreboard' : meas_stall_long_scoreboard,
        'stall_mio_throttle' : meas_stall_mio_throttle
    })
    return df

def post_process_kernel_metrics(df_kernel_metrics: pd.DataFrame) -> pd.DataFrame:
    if len(df_kernel_metrics) == 0:
        return df_kernel_metrics

    df_grouped = df_kernel_metrics.groupby('kernel').aggregate({
        'latency': 'mean',
        'sp_gflops': 'sum',
        'sp_to_peak_pct': 'mean',
        'dp_gflops': 'sum',
        'dp_to_peak_pct': 'mean',
        'compute_thr': 'mean',
        'memory_thr': 'mean',
        'l1_tex_cache_thr': 'mean',
        'l2_cache_thr': 'mean',
        'dram_thr': 'mean',
        'occ_theor': 'mean',
        'act_warp_sm_theor': 'mean',
        'occ_achieved': 'mean',
        'act_warp_sm_achieved': 'mean',
        'blk_lim_regs': 'mean',
        'blk_lim_sh_mem': 'mean',
        'blk_lim_warps': 'mean',
        'blk_lim_sm': 'mean',
        'grid_size': 'mean',
        'blk_size': 'mean',
        'sm_count': 'mean',
        'gc': 'mean',
        'warp_cycles_per_issued_inst' : 'mean',
        'avg_act_threads_per_warp' : 'mean',
        'avg_not_pred_threads_per_warp' : 'mean',
        'warp_cycles_per_Exec_inst' : 'mean',
        'stall_drain' : 'mean',
        'stall_imc_miss' : 'mean',
        'stall_barrier' : 'mean',
        # 'stall_gmma' : 'mean',
        'stall_branch_resolving' : 'mean',
        'stall_membar' : 'mean',
        'stall_short_scoreboard'  : 'mean', 
        'stall_sleeping' : 'mean',
        'stall_wait' : 'mean',
        'stall_no_instr' : 'mean',
        'stall_math_pipe_throttle' : 'mean',
        'stall_text_throttle' : 'mean',
        'stall_lg_throttle' : 'mean',
        'stall_dispath_stall' : 'mean',
        'stall_misc' : 'mean',
        'stall_not_selected' : 'mean',
        'stall_selected' : 'mean',
        'stall_long_scoreboard' : 'mean',
        'stall_mio_throttle' : 'mean'
    }).reset_index()
    df_grouped['count'] = df_kernel_metrics.groupby('kernel').size().values

    return df_grouped
    








                
        
