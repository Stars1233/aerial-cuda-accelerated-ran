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

from os import path
import pandas as pd
import numpy as np
from datetime import datetime

month_map = {
    "Jan": 1, "Feb": 2, "Mar": 3, "Apr": 4,
    "May": 5, "Jun": 6, "Jul": 7, "Aug": 8,
    "Sep": 9, "Oct": 10, "Nov": 11, "Dec": 12
}

def get_monitor_cpu(parent_folder):
    monitor_cpu_file = path.join(parent_folder, 'monitor_cpu_cores.txt')
    if not path.isfile(monitor_cpu_file):
        return None
    else:
        with open(monitor_cpu_file) as f:
            return f.read()

def get_phy_monitor_cpu(scenario_folder):
    return get_monitor_cpu(path.join(scenario_folder, 'cuphy'))

def get_mac_monitor_cpu(scenario_folder):
    return get_monitor_cpu(path.join(scenario_folder, 'mac'))

def get_ru_monitor_cpu(scenario_folder):
    return get_monitor_cpu(path.join(scenario_folder, 'ru'))

def read_usage_consumption(content: str):
    if content is None:
        return None
    
    df_map = {
        'timestamp': [],
        'thread_tid': [],
        'thread_name': [],
        'thread_policy': [],
        'thread_affinity': [],
        'thread_prio': [],
        'thread_cpu_usage': [],
        'thread_mem_usage': []
    }
    
    timestamp = None
    for line in content.splitlines():
        if 'CPU CORE INFO FOR' in line:
            # example
            # CPU CORE INFO FOR cuphycontroller_scf - Wed Jun 12 01:30:52 UTC 2024
            tokens = line.split()
            year = int(tokens[-1])
            time_str = tokens[-3]
            day = int(tokens[-4])
            month = month_map[tokens[-5]]
            datetime_str = f'{year}-{month:02d}-{day:02d} {time_str}'
            datetime_obj = datetime.strptime(datetime_str, '%Y-%m-%d %H:%M:%S')
            timestamp = datetime_obj.timestamp()
        elif 'TS' in line or 'FF' in line:
            # example
            # 2570    2570 phy_main        TS   23  20 30.4  1.4 110930688 8598528
            tokens = line.split()
            df_map['timestamp'].append(timestamp)
            df_map['thread_tid'].append(int(tokens[1]))
            df_map['thread_name'].append(tokens[2])
            df_map['thread_policy'].append(tokens[3])
            df_map['thread_affinity'].append(int(tokens[4]))
            df_map['thread_prio'].append(int(tokens[5]))
            df_map['thread_cpu_usage'].append(float(tokens[6])/100)
            df_map['thread_mem_usage'].append(float(tokens[7])/100)     
    return pd.DataFrame(df_map)

def post_process_resource_consumption(df: pd.DataFrame):
    if df is None:
        return None

    # Average across timestamp    
    df = df.groupby('thread_name')[['thread_affinity', 'thread_prio', 'thread_cpu_usage']].aggregate({
        'thread_affinity': lambda x: x.iloc[0],
        'thread_prio': lambda x: x.iloc[0],
        'thread_cpu_usage': np.mean
    }).sort_values(by='thread_cpu_usage', ascending=False).reset_index()

    # Devised this notion of thread_category so to group together threads
    # for visualization purposes.
    df.loc[:, 'thread_category'] = df.thread_name
    for thread_name in ['UlPhyDriver', 'DlPhyDriver', 'ul_core', 'dl_proc']:
            mask = df.thread_name.str.contains(thread_name)
            if df[mask].shape[0] > 0:
                df.loc[mask, ['thread_category']] = [f'{thread_name}']

    # Find CPUs low priority threads (prio=20) are running on.
    low_priority_cpus = df[df.thread_prio==20].thread_affinity.unique()
 
    # Set thread_category for threads that are running on those CPUs
    # These threads can be set low or even high priority but all of these 
    # threads will be grouped together.
    df.loc[df.thread_affinity.isin(low_priority_cpus), 'thread_category'] = 'other'

    # Find the CPUs that high priority threads are running on (thread prio < 0) and they don't belong to the low priority CPUs
    high_priority_affinity = df[(df.thread_prio < 0) & (~df.thread_affinity.isin(low_priority_cpus))].groupby('thread_affinity').count().reset_index()
    
    # Among these CPUs, get the ones that more than one high priority threads are running 
    high_priority_threads_sharing_cpu = high_priority_affinity[high_priority_affinity.thread_name > 1].thread_affinity.unique()          
    
    # For the threads that are runining on these CPUs, set their category
    for high_priority_cpu in high_priority_threads_sharing_cpu:    
        mask = df.thread_affinity==high_priority_cpu
        df.loc[mask, 'thread_category'] = df[mask].iloc[0].thread_category

    df_grouped = df.groupby(['thread_category']).aggregate({
        'thread_cpu_usage': [('', np.sum)],
        'thread_affinity': [('', 'nunique'), ('cpu_list', lambda x: ','.join(map(str,sorted(x.unique()))))],
        'thread_name': [('',','.join)]
    })
    df_grouped.columns = ['_'.join(col).strip('_') for col in df_grouped.columns.values]
    df_grouped = df_grouped.reset_index().sort_values(by='thread_cpu_usage', ascending=False)

    show_cols = 6
    if (df_grouped.shape[0] > show_cols):
        # If there are more than 6 categories, just add statistics to the 6th one
        # This is done for visualization purposes
        df_rest = df_grouped.iloc[show_cols:]        
        df_grouped.iloc[show_cols-1, df_grouped.columns.get_loc('thread_cpu_usage')] += df_rest.thread_cpu_usage.sum()
        df_grouped.iloc[show_cols-1, df_grouped.columns.get_loc('thread_affinity')] += df_rest.thread_affinity.sum()
        df_grouped = df_grouped.iloc[:show_cols]
    df_grouped.loc[:, 'cpu_unused'] = df_grouped.thread_affinity - df_grouped.thread_cpu_usage
    return df_grouped

def read_phy_cpu_consumption(scenario_folder):
    content = get_phy_monitor_cpu(scenario_folder)
    phy_cpu_consumption = read_usage_consumption(content)
    post_processed_df = post_process_resource_consumption(phy_cpu_consumption)

    phy_cpu_consumption['timestamp'] = phy_cpu_consumption['timestamp'] - phy_cpu_consumption['timestamp'].min()
    phy_cpu_consumption['timestamp'] = phy_cpu_consumption['timestamp'].astype(np.int32)
    phy_cpu_consumption = phy_cpu_consumption.groupby(['timestamp', 'thread_name', 'thread_affinity']).aggregate({
        'thread_policy': 'first',
        'thread_prio': 'first',
        'thread_cpu_usage': 'sum',
        'thread_mem_usage': 'sum'
    }).reset_index()
    return phy_cpu_consumption, post_processed_df

def read_mac_cpu_consumption(scenario_folder):
    content = get_mac_monitor_cpu(scenario_folder)
    return post_process_resource_consumption(read_usage_consumption(content))

def read_ru_cpu_consumption(scenario_folder):
    content = get_ru_monitor_cpu(scenario_folder)
    return post_process_resource_consumption(read_usage_consumption(content))
